# Voice RAG Chatbot

This Jupyter Notebook demonstrates a Voice RAG (Retrieval-Augmented Generation) Chatbot that allows users to interact with a large language model (Mistral-7B) using voice input and receive voice responses. The chatbot retrieves relevant information from uploaded PDF documents to provide context-aware answers.

## Features

- Upload multiple PDF files to create a knowledge base
- Speak your question using the microphone
- Transcribe the audio input into text using the Whisper ASR model
- Retrieve relevant context from the uploaded PDFs using Qdrant vector store
- Generate a response using the Mistral-7B language model
- Convert the generated response into audio using the WheelSpeech TTS model
- Play the audio response directly in the notebook

## Requirements

- Python 3.x
- Transformers library
- Torch library
- Langchain library
- Gradio library
- Whisper library
- WhisperSpeech library

## Requirements

Download all the dependences

```
pip install -r requirements.txt
```

## Usage

1. Run the Jupyter Notebook and execute the code cells.
2. Upload one or more PDF files using the file uploader component.
3. Click the "Submit Audio" button and speak your question into the microphone.
4. The chatbot will transcribe your audio input, retrieve relevant context from the uploaded PDFs, generate a response, and play the audio response.
5. To clear all inputs and outputs, click the "Clear All" button.

## Functions

- `load_llm()`: Loads the Mistral-7B language model for text generation.
- `embeddings_model()`: Loads the all-mpnet-base-v2 sentence transformer for generating embeddings.
- `text_splitter()`: Creates a RecursiveCharacterTextSplitter for splitting long documents into chunks.
- `add_pdfs_to_vectorstore(files)`: Processes uploaded PDF files and adds them to the Qdrant vector store.
- `answer_query(message)`: Retrieves relevant context based on the user's question and generates a response using the language model.
- `generate_and_play_audio(text)`: Converts the generated response into audio using the WheelSpeech TTS model and plays it.
- `transcribe(audio)`: Transcribes the user's audio input into text using the Whisper ASR model.

## Gradio UI Components

- `upload_files`: File uploader for uploading PDF files.
- `success_msg`: Text component to display the success message after uploading files.
- `audio_inp`: Audio component for capturing user's voice input.
- `trans_out`: Textbox component to display the transcribed text.
- `btn_audio`: Button component to trigger audio transcription.
- `model_response`: Textbox component to display the generated response from the chatbot.
- `audio_out`: Audio component to play the generated audio response.
- `clear_btn`: Button component to clear all inputs and outputs.

## Notes

- The chatbot uses the Mistral-7B language model, which requires significant computational resources. Ensure you have sufficient memory and a compatible GPU.
- The chatbot relies on the uploaded PDF files for providing context-aware responses. Make sure to upload relevant documents before asking questions.
- The audio transcription and generation may take some time depending on the length of the input and the complexity of the response.

Feel free to explore and interact with the Voice RAG Chatbot to experience voice-based conversational AI with retrieval-augmented generation capabilities!