import pandas as pd
import os
import joblib
import torch
from fastapi import FastAPI, HTTPException
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os
from training import tfidf_class
from analysis_and_cleaning import anaylizer_cleaner
from pydantic import BaseModel
import torch

app = FastAPI()

class ReviewRequest(BaseModel):
    review: str


@app.get("/")
async def root():
    return {"message": "FastAPI is running!"}


#Load model
model_path = "meta-llama/Llama-3.2-1B-Instruct"

model = AutoModelForCausalLM.from_pretrained(model_path, use_auth_token="")
tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token="")

#Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",     #for RAM optimization
    trust_remote_code=True  
)

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)


#Summarizer pipeline
LLM_summarizer = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype="auto",
    device_map="auto",  
)


#ML and TD-IDF models
classifier_model_path = "models/Logistic_Regression.pkl"
model = joblib.load(classifier_model_path)

vectorizer_path = "models/tfidf_vectorizer.pkl"
vectorizer = joblib.load(vectorizer_path)


label_type = {0: 'NEGATIVE', 1: 'POSITIVE'}


@app.post("/add_review_and_analyze")
async def add_review_and_analyze(review_request: ReviewRequest):

    most_reviewed_topics = pd.read_csv("datasets/similarity_reviews_groups.csv") 
    review = review_request.review

    new_review_cleaned = anaylizer_cleaner.combined_preprocessing(review) 
    new_review_tfidf = vectorizer.transform([" ".join(new_review_cleaned)])
    
    
    prediction = model.predict(new_review_tfidf)
    
    label = label_type[prediction[0]]

    last_review = pd.DataFrame({
        "cleaned_review": new_review_cleaned,
        "label" : label
    })

    #Combine dataset and new review
    most_reviewed_topics = pd.concat([most_reviewed_topics, last_review], ignore_index=True)

    #Return all values into string
    most_reviewed_topics['Review'] = most_reviewed_topics['Review'].astype(str)

    #Concatenate the dataset with the last comment and convert it to string
    single_sentence = " ".join(most_reviewed_topics['Review'])

    #Clean unnecessary characters
    single_sentence = single_sentence.replace('[', '').replace(']', '').replace('\\', '').replace("'", '').replace(',', '')

    messages = [
        {"role": "system", "content":   "You are a summarization chatbot who always summarize."
                                        "Summarize and write in a single short paragraph about Spotify reviews."
                                        "Don't write my messages in output."},
        {"role": "user", "content": single_sentence},
    ]
    
    outputs = LLM_summarizer(
        messages,
        max_new_tokens=200,
    )

    return {"Conclusion": outputs[0]["generated_text"][-1], "label": label}