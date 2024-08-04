import json
import re
import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import numpy as np
import spacy

# Load the spaCy model for NER
nlp = spacy.load("en_core_web_sm")

# Function to load the JSON file
def load_transcript(json_file):
    with open('transcript.json', 'r') as file:
        transcript = json.load(file)
    return transcript

# Function to extract dialogues from the transcript
def extract_dialogues(transcript):
    dialogues = []
    for entry in transcript:
        text = entry.get('text')
        if text:
            sentences = re.split(r'(?<=[.!?]) +', text)
            dialogues.extend(sentences)
        else:
            print(f"Missing or empty 'text' in entry: {entry}")
    return dialogues

# Function for text cleaning and preprocessing
def clean_and_preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    return " ".join(tokens)

# Function to find the most relevant section
def find_relevant_section(query, sections, embedding_model, section_embeddings):
    query_embedding = embedding_model.encode([query], convert_to_tensor=True)

    if section_embeddings.shape[0] == 0:
        return "No relevant section found."

    similarities = util.pytorch_cos_sim(query_embedding, section_embeddings)[0]
    most_similar_idx = np.argmax(similarities)
    return sections[most_similar_idx]

# Function to get the answer from the relevant section
def get_answer_from_section(question, context, qa_model):
    result = qa_model(question=question, context=context)
    if len(result['answer'].split()) < 3:
        return "Sorry, I couldn't find a precise answer in the transcript."
    return result['answer']

# Load and preprocess transcript
transcript = load_transcript('processed_transcript.json')
dialogues = extract_dialogues(transcript)

if not dialogues:
    st.error("No dialogues found in the transcript.")
else:
    cleaned_dialogues = [clean_and_preprocess_text(dialogue) for dialogue in dialogues]

    # Load pre-trained models
    qa_model = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

    embedding_model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

    # Compute embeddings for transcript sections
    section_embeddings = embedding_model.encode(cleaned_dialogues, convert_to_tensor=True)

    # User input for the question
    question = st.text_input("Enter your question:")
    if question:
        relevant_section = find_relevant_section(question, cleaned_dialogues, embedding_model, section_embeddings)
        answer = get_answer_from_section(question, relevant_section, qa_model)
        st.write(answer)
