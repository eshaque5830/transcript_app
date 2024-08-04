import json
import spacy
import re

# Load the spaCy model for NER
nlp = spacy.load("en_core_web_sm")

# Function to load the JSON file
def load_transcript(json_file):
    with open(json_file, 'r') as file:
        transcript = json.load(file)
    return transcript

# Function to extract dialogues from the transcript
def extract_dialogues(transcript):
    dialogues = []
    for entry in transcript:
        dialogues.append(entry['text'])
    return dialogues

# Function for text cleaning and preprocessing
def clean_and_preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove unnecessary characters (punctuation, special characters, etc.)
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenize the text (split text into words)
    tokens = text.split()
    
    return " ".join(tokens)

# Function to perform NER and extract entities
def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Example usage
if __name__ == "__main__":
    # Load the JSON file
    transcript = load_transcript('transcript.json')
    
    # Extract and preprocess dialogues
    dialogues = extract_dialogues(transcript)
    cleaned_dialogues = [clean_and_preprocess_text(dialogue) for dialogue in dialogues]
    
    # Perform NER on each dialogue
    for dialogue in cleaned_dialogues:
        entities = extract_entities(dialogue)
        print(f"Dialogue: {dialogue}")
        print(f"Entities: {entities}\n")
