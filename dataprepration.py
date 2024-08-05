#streamlit 3.10.0
import spacy
import json
import re
import os

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    """Remove punctuation and lowercase the text."""
    return re.sub(r'[^\w\s]', '', text.lower())

def process_line(line):
    """Process a single line of text and return a structured dictionary."""
    # Check if the line contains a colon
    if ":" not in line:
        print(f"Skipping line due to missing colon: {line}")
        return None

    try:
        # Split the line into speaker and text
        speaker, original_text = line.split(":", 1)
    except ValueError as e:
        print(f"Error splitting line: {line}\n{e}")
        return None

    speaker = speaker.strip()
    original_text = original_text.strip()

    # Clean the text
    cleaned_text = clean_text(original_text)

    # Tokenize the text and extract entities using spaCy
    doc = nlp(cleaned_text)
    tokens = [token.text for token in doc]
    entities = [[ent.text, ent.label_] for ent in doc.ents]

    # Create the structured dictionary
    return {
        "speaker": speaker,
        "original_text": original_text,
        "cleaned_text": cleaned_text,
        "tokens": tokens,
        "entities": entities
    }

def process_transcript(transcript_text):
    """Process the entire transcript into structured JSON format."""
    # Split the transcript into lines and process each line
    lines = transcript_text.strip().splitlines()
    return [process_line(line) for line in lines]

def main(input_file):
    """Read the plain text file, process it, and save the output as JSON."""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Read the plain text file
    with open(input_file, 'r') as file:
        transcript_text = file.read()

    # Process the transcript
    structured_transcript = process_transcript(transcript_text)

    # Define the output file path
    output_file = os.path.join(script_dir, 'processed_transcript.json')

    # Save the output to a JSON file in the same directory as the script
    with open(output_file, 'w') as file:
        json.dump(structured_transcript, file, indent=4)

    print(f"Processed transcript saved to {output_file}")

# Specify the input file name
input_file = 'original_text.txt'  # Replace with your input file name

# Run the main function
main(input_file)
