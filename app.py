import json
import torch
import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# Function to load the processed transcript JSON file
def load_transcript(json_file):
    with open('processed_transcript.json', 'r') as file:
        transcript = json.load(file)
    return transcript

# Function to extract cleaned dialogues from the processed transcript
def extract_cleaned_dialogues(transcript):
    cleaned_dialogues = []
    for entry in transcript:
        cleaned_text = entry.get('cleaned_text')
        if cleaned_text:
            cleaned_dialogues.append(cleaned_text)
        else:
            print(f"Missing or empty 'cleaned_text' in entry: {entry}")
    return cleaned_dialogues

# Function to find the most relevant section
def find_relevant_section(query, sections, embedding_model, section_embeddings, threshold=0.3):
    query_embedding = embedding_model.encode([query], convert_to_tensor=True)

    if section_embeddings.shape[0] == 0:
        return "No relevant section found."

    # Calculate similarities between query and sections
    similarities = util.pytorch_cos_sim(query_embedding, section_embeddings)[0]
    
    # Print similarity scores for debugging
    print("Similarity scores:", similarities.tolist())

    most_similar_idx = torch.argmax(similarities).item()
    
    if similarities[most_similar_idx] < threshold:
        return "Sorry, I couldn't find a precise answer in the transcript."
    
    return sections[most_similar_idx]

# Function to get the answer from the relevant section
def get_answer_from_section(question, context, qa_model):
    result = qa_model(question=question, context=context)
    if len(result['answer'].split()) < 3:
        return "Sorry, I couldn't find a precise answer in the transcript."
    return result['answer']

# Load and preprocess the transcript
transcript = load_transcript('processed_transcript.json')
cleaned_dialogues = extract_cleaned_dialogues(transcript)

# Load a pre-trained sentence transformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for all dialogues
section_embeddings = embedding_model.encode(cleaned_dialogues, convert_to_tensor=True)

# Load the QA model
qa_model = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Streamlit UI
st.title("Doctor-Patient Q&A App by M.Ishaque")
query = st.text_input("Ask a question based on the transcript:")
if query:
    relevant_section = find_relevant_section(query, cleaned_dialogues, embedding_model, section_embeddings)
    if "Sorry" not in relevant_section:
        answer = get_answer_from_section(query, relevant_section, qa_model)
        st.write(f"Answer: {answer}")
    else:
        st.write(relevant_section)
