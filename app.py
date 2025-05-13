from flask import Flask, render_template, request, jsonify
import os
import fitz
import re
import numpy as np
import faiss
import pickle
import tkinter as tk
from tkinter import filedialog
from werkzeug.utils import secure_filename
from zhipuai import ZhipuAI
from scraper import WechatScraper


client = ZhipuAI(api_key="your_api_key")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SCRAPED_FOLDER'] = 'scraped'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SCRAPED_FOLDER'], exist_ok=True)

# Initialize FAISS index
dimension = 384
index = faiss.IndexFlatL2(dimension)
sentences = []


def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text


def get_embedding(text, model="embedding-3"):
    response = client.embeddings.create(
        model=model,
        input=[text]
    )
    return response.data[0].embedding


def split_text_to_sentences(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def create_faiss_index(sentences):
    embeddings = []
    for sentence in sentences:
        embedding = get_embedding(sentence)
        embeddings.append(embedding)

    embeddings = np.array(embeddings).astype('float32')

    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, sentences


def retrieve_similar_sentences(query, index, sentences, k=20):
    query_embedding = np.array(get_embedding(query)).astype('float32')
    distances, indices = index.search(query_embedding.reshape(1, -1), k)
    similar_sentences = [sentences[i] for i in indices[0]]
    return similar_sentences


def generate_answer(query, context):
    messages = [
        {"role": "system",
         "content": "You are an AI assistant specialized in answering questions based on provided context."},
        {"role": "user", "content": f"Context: {' '.join(context)}\n\nQuestion: {query}\n\nAnswer:"}
    ]
    response = client.chat.completions.create(model="glm-4", messages=messages, max_tokens=300)
    return response.choices[0].message.content.strip()


def choose_project_folder():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Select Project Folder")
    if not folder_path:
        print("No folder selected. Exiting process.")
        exit()
    return folder_path


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    global index, sentences

    keyword = request.form.get('keyword')
    files = request.files.getlist('file')
    result = {"status": "success"}

    # Handle WeChat scraping if keyword is provided
    if keyword:
        scraper = WechatScraper()
        try:
            links = scraper.search(keyword)
            articles = []
            for link in links:
                content = scraper.get_content(link)
                articles.append(content)

                # Save scraped content
                title = content["title"].replace("/", "-")
                text_filename = f"{title}.txt"
                text_filepath = os.path.join(app.config['SCRAPED_FOLDER'], text_filename)
                with open(text_filepath, 'w', encoding='utf-8') as f:
                    f.write(content["content"])

                # Split text into sentences and add to the global list
                sentences.extend(split_text_to_sentences(content["content"]))

            result["articles"] = articles
        except Exception as e:
            result["scrape_error"] = str(e)
        finally:
            scraper.close()

    # Handle PDF uploads
    uploaded_files = []
    for file in files:
        if file and file.filename.endswith('.pdf'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Extract text from PDF
            text = extract_text_from_pdf(filepath)

            # Save the extracted text
            text_filename = os.path.splitext(filename)[0] + ".txt"
            text_filepath = os.path.join(app.config['UPLOAD_FOLDER'], text_filename)
            with open(text_filepath, 'w', encoding='utf-8') as f:
                f.write(text)

            # Split text into sentences and add to the global list
            sentences.extend(split_text_to_sentences(text))

            uploaded_files.append({"filename": filename, "text_file": text_filename})

    if uploaded_files:
        result["uploaded_files"] = uploaded_files

    # Create or update FAISS index
    if sentences:
        index, sentences = create_faiss_index(sentences)
        # Save the FAISS index and sentences to disk in the FAISS files directory
        faiss.write_index(index, os.path.join('FAISS files', "faiss_index.bin"))
        with open(os.path.join('FAISS files', "sentences.pkl"), "wb") as f:
            pickle.dump(sentences, f)

    return jsonify(result)


@app.route('/ask', methods=['POST'])
def ask_question():
    global index, sentences

    question = request.json.get('question')

    # Search FAISS index for relevant documents
    if not index or not sentences:
        # Load from disk if not initialized
        try:
            # Load from FAISS files directory
            index = faiss.read_index(os.path.join('FAISS files', "faiss_index.bin"))
            with open(os.path.join('FAISS files', "sentences.pkl"), "rb") as f:
                sentences = pickle.load(f)
        except:
            return jsonify({"answer": "No context available. Please upload files or scrape content first."})

    similar_sentences = retrieve_similar_sentences(question, index, sentences)

    # Generate answer using the retrieved context
    answer = generate_answer(question, similar_sentences)

    return jsonify({"answer": answer})


@app.route('/initialize', methods=['POST'])
def initialize():
    global index, sentences

    # Collect all PDF files from the folder
    folder_path = choose_project_folder()
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]

    sentences = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        doc = fitz.open(pdf_path)
        text = extract_text_from_pdf(doc)
        sentences.extend(split_text_to_sentences(text))

    # Create FAISS index
    index, sentences = create_faiss_index(sentences)

    # Save the FAISS index and sentences to disk in the FAISS files directory
    faiss.write_index(index, os.path.join('FAISS files', "faiss_index.bin"))
    with open(os.path.join('FAISS files', "sentences.pkl"), "wb") as f:
        pickle.dump(sentences, f)

    return jsonify({"status": "Initialization complete"})

if __name__ == '__main__':
    app.run(debug=True)
