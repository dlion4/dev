from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re
# Initialize FastAPI app
app = FastAPI()

# Load Hugging Face pipelines
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

class TextInput(BaseModel):
    text: str



def extract_keywords(text, top_n=5):
    # Tokenize the text and calculate TF-IDF scores
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    scores = np.array(tfidf_matrix.sum(axis=0)).flatten()

    # Get top N keywords based on TF-IDF scores
    top_indices = scores.argsort()[-top_n:][::-1]
    keywords = [(feature_names[i], scores[i]) for i in top_indices]
    
    weighted_keywords = [f"{score:.3f}*'{keyword}'" for keyword, score in keywords]
    return " + ".join(weighted_keywords)


@app.get("/")
def check_application_health()->str:
    return "everything is file and upto date"

# Summarization Endpoint
@app.get("/summarize")
def summarize_text(input:str="This is a test application to check on the api integration on the app without having to install the models on my system but use the free github workspaces"):
    try:
        summary = summarizer(input, max_length=150, min_length=40, do_sample=False)
        print(summary[0]['summary_text'])
        return {"summary": summary[0]['summary_text']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

TEST_MESSAGE = """
Users with a qualifying Gemini for Google Workspace add-on subscription can now upload Google Sheets, 
Google Docs, CSVs and PDFs, as well as Excel and Word files from Google Drive or your device. With Document Upload,
business users can elevate research and writing tasks, and identify trends within those documents.
With Data Analysis, business users can process and explore data for deeper insights as well as create presentation-ready charts.
"""
@app.get("/extract_topic")
def extract_topic(input:str= TEST_MESSAGE):
    try:
        summary = summarizer(input, do_sample=False)
        summarized_text = summary[0]['summary_text']
        main_topic = extract_highest_score_word(extract_keywords(summarized_text))
        return {"topic": main_topic}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def extract_highest_score_word(topic_string):
    """
    Extracts the word with the highest score from a formatted topic string.
    Example input: "0.200*'intelligence' + 0.180*'machine' + 0.150*'learning'"
    """
    matches = re.findall(r"([0-9.]+)\*'(\w+)'", topic_string)
    if matches:
        highest_score_pair = max(matches, key=lambda x: float(x[0]))
        return highest_score_pair[1]
    return None