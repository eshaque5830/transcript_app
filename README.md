This app use processed Doctor and Patient transcript in .JSON i have also made a user interface through streamlit for interaction.
1- dataprepration.py: convert text from plan text to json format.
  
                      Here's a summary of the code:
                      
                      Setup:
                      
                      Imports necessary libraries: spacy for NLP, json for handling JSON data, re for text cleaning, and os for file operations.
                      Loads a spaCy language model for processing English text.
                      Functions:
                      
                      clean_text(text): Removes punctuation and converts text to lowercase.
                      process_line(line): Parses each line of the transcript, splitting it into speaker and text. Cleans and tokenizes the text, extracts named entities, and structures this data in a dictionary.
                      process_transcript(transcript_text): Processes the entire transcript by applying process_line to each line and collects the results.
                      Main Execution:
                      
                      main(input_file): Reads a plain text file containing the transcript, processes it, and saves the structured data to a JSON file named processed_transcript.json.
                      Script Execution:
                      
                      Specifies the name of the input file and runs the main function to process the transcript.
                      Summary
                      The script reads a transcript from a text file, processes each line to extract and clean text, performs NLP analysis, and saves the processed information in JSON format.
2- app.py:  
                      Here's a summary of the provided code:

                      Setup:Imports libraries: json for handling JSON files, torch for tensor operations, streamlit for creating the web app, transformers for the QA model, and sentence_transformers for generating sentence embeddings.
                      Functions:
                      
                      load_transcript(json_file): Loads the processed transcript JSON file.
                      extract_cleaned_dialogues(transcript): Extracts cleaned dialogue texts from the transcript.
                      find_relevant_section(query, sections, embedding_model, section_embeddings, threshold=0.3): Finds the most relevant section of the transcript based on the query using sentence embeddings and cosine similarity.
                      get_answer_from_section(question, context, qa_model): Retrieves an answer from the relevant section using a QA model.
                      Processing:
                      
                      Loads and preprocesses the transcript from processed_transcript.json.
                      Uses SentenceTransformer to generate embeddings for all cleaned dialogues.
                      Loads a pre-trained QA model from the transformers library.
                      Streamlit UI:
                      
                      Displays the title of the app and a sidebar with predefined questions.
                      Allows users to ask a question from the predefined list or enter their own question.
                      Processes the input question to find the most relevant section of the transcript and gets the answer using the QA model.
                      Displays the answer or a message if no relevant section is found.
                      Summary
                      The code sets up a Streamlit web app that lets users ask questions about a doctor-patient transcript. It processes the transcript to find the most relevant section based on user queries, extracts answers using a pre-trained QA model, and displays the results in the app.
