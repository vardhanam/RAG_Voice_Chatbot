from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)

import torch
import torch.nn.functional as F

import os

from langchain.llms import HuggingFacePipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings

import gradio as gr

import whisper
from whisperspeech.pipeline import Pipeline

import uuid


def load_llm():
    # Loading the Mistral Model
    model_name = 'mistralai/Mistral-7B-Instruct-v0.2'
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
    )

    # Building a LLM text-generation pipeline
    text_generation_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        temperature=0.2,
        repetition_penalty=1.1,
        return_full_text=True,
        max_new_tokens=1024,
        device_map='auto',
    )

    return text_generation_pipeline


def text_splitter():
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter


# Load the Whisper model for transcription
whisper_model = whisper.load_model("base")
# Load the Whisper Speech model for audio generation
whisper_speech_model = Pipeline(s2a_ref='collabora/whisperspeech:s2a-q4-tiny-en+pl.model')
# Load the LLM model
llm = load_llm()
# Load the text splitter
textsplitter = text_splitter()


def add_pdfs_to_vectorstore(files):
    saved_files_count = 0
    documents = []
    for file_path in files:
        file_name = os.path.basename(file_path)  # Extract the filename from the full path
        if file_name.lower().endswith('.pdf'):  # Check if the file is a PDF
            saved_files_count += 1
            loader_temp = PyPDFLoader(file_path)
            docs_temp = loader_temp.load_and_split(text_splitter=textsplitter)
            for doc in docs_temp:
                # Replace all occurrences of '\n' with a space ' '
                doc.page_content = doc.page_content.replace('\n', ' ')
            documents = documents + docs_temp
        else:
            print(f"Skipping non-PDF file: {file_name}")

    global qdrant

    # Create a Qdrant vectorstore from the documents
    qdrant = Qdrant.from_documents(
        documents,
        HuggingFaceEmbeddings(),
        location=":memory:",
    )

    return f"Added {saved_files_count} PDF file(s) to vectorstore/ You can begin voice chat"


def answer_query(message):
    # Perform a similarity search on the vectorstore to find relevant context
    context_docs = qdrant.similarity_search(message, k=10)
    context = ' '.join(doc.page_content for doc in context_docs)

    # Create a template for the question-answering prompt
    template = f"""Answer the question based only on the following context:
        {context}

        Question: {message}
    """

    # Generate a response using the LLM
    result = llm(template)

    # Extract the answer from the generated response
    answer = result[0]["generated_text"].replace(template, '')

    return answer


def transcribe(audio):
    # Transcribe the audio using the Whisper model
    result = whisper_model.transcribe(audio)
    return result["text"]


def generate_and_play_audio(text, filename):
    # Construct the directory and filename
    directory = os.path.join(os.getcwd(), 'audio_output')
    file_location = os.path.join(directory, filename)

    # Ensure that the directory exists
    os.makedirs(directory, exist_ok=True)

    # Generate the audio file from text and save to the specified location
    whisper_speech_model.generate_to_file(file_location, text, lang='en', cps=15)

    # Return the location of the saved audio file for playback
    return file_location


from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/upload_pdfs', methods=['POST'])
def upload_pdfs():
    # Expecting a JSON payload with an array of file paths under the key 'file_paths'
    data = request.get_json()
    file_paths = data.get('file_paths') if data else None

    if not file_paths or not isinstance(file_paths, list):
        return jsonify({'error': 'No file paths provided or file paths is not a list'}), 400

    # Assuming add_pdfs_to_vectorstore is a function that processes the list of file paths
    try:
        result = add_pdfs_to_vectorstore(file_paths)
        return jsonify({'message': result}), 200
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': 'An error occurred while processing the file paths'}), 500


@app.route('/process_audio', methods=['POST'])
def process_audio():
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    audio_path = data.get('audio_path')
    output_filename = data.get('output_filename')

    if not audio_path:
        return jsonify({'error': 'No audio path provided'}), 400
    if not output_filename:
        return jsonify({'error': 'No output filename provided'}), 400

    # Assuming the audio file exists at the given path and is accessible by the server
    try:
        transcription = transcribe(audio_path)
        response_text = answer_query(transcription)
        audio_output_path = generate_and_play_audio(response_text, output_filename)

        # Get the relative path of the audio output file
        relative_audio_output_path = os.path.relpath(audio_output_path, os.getcwd())

        return jsonify({
            'transcription': transcription,
            'response_text': response_text,
            'audio_output': audio_output_path
        })
    except IOError:
        return jsonify({'error': 'Audio file not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)