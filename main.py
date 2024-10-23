from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import json
import os

app = FastAPI()

# Ruta del archivo JSON donde se guardarán las preguntas y respuestas seleccionadas
json_file = 'user_feedback.json'

# Descargar y preparar modelo BETO para question-answering en español
qa_pipeline = pipeline('question-answering', model="mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es")

# Función para cargar preguntas y respuestas guardadas desde el archivo JSON
def load_feedback():
    if os.path.exists(json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        return []

# Función para guardar nuevas preguntas y respuestas en el archivo JSON
def save_feedback(new_entry):
    data = load_feedback()
    data.append(new_entry)
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# Cargar preguntas y respuestas previamente guardadas en el archivo JSON
feedback_data = load_feedback()

# Send a GET request to the website and get the HTML response
url = "https://www.billetesymonedas.cl/home/preguntasfrecuentes"
response = requests.get(url)

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(response.content, 'html.parser')

# Find all the questions and answers on the page
questions = []
answers = []

questionsTemp = soup.find_all('p', class_='pregunta')
answersTemp = soup.find_all('blockquote', class_='respuesta')

# If there's a mismatch, use the smaller number
min_len = min(len(questionsTemp), len(answersTemp))

# Guardar preguntas y respuestas en español tal cual están en la página
for i in range(min_len):
    question_es = questionsTemp[i].text.strip()
    answer_es = answersTemp[i].text.strip()
    questions.append(question_es)
    answers.append(answer_es)

# Añadir las respuestas guardadas del JSON al contexto
for entry in feedback_data:
    questions.append(entry['question'])
    answers.append(entry['chosen_answer'])

# Pydantic model for the input
class QuestionRequest(BaseModel):
    question: str

class FeedbackRequest(BaseModel):
    question: str
    chosen_answer: str

# Función para obtener todas las respuestas rankeadas
def get_ranked_answers(question, context_list):
    ranked_answers = []
    for context in context_list:
        result = qa_pipeline(question=question, context=context)
        ranked_answers.append({'answer': context, 'score': result['score']})
    ranked_answers = sorted(ranked_answers, key=lambda x: x['score'], reverse=True)
    return ranked_answers

# Diccionario para almacenar el estado de las respuestas mostradas por sesión
session_data: Dict[str, Any] = {}

@app.post("/ask")
def ask_question(request: QuestionRequest):
    question = request.question
    ranked_answers = get_ranked_answers(question, answers)
    session_data[question] = ranked_answers
    top_5_answers = ranked_answers[:5]
    return {
        "question": question,
        "answers": [{"answer": ans["answer"], "score": ans["score"]} for ans in top_5_answers],
        "next_page": 1
    }

@app.post("/ask/next")
def ask_next_page(request: QuestionRequest):
    question = request.question
    if question not in session_data:
        raise HTTPException(status_code=400, detail="Pregunta no encontrada. Realiza una nueva pregunta primero.")
    ranked_answers = session_data[question]
    current_page = session_data.get(f"{question}_page", 1)
    start_index = current_page * 5
    end_index = start_index + 5
    next_answers = ranked_answers[start_index:end_index]
    session_data[f"{question}_page"] = current_page + 1
    if not next_answers:
        return {"message": "No hay más respuestas disponibles para esta pregunta."}
    return {
        "question": question,
        "answers": [{"answer": ans["answer"], "score": ans["score"]} for ans in next_answers],
        "next_page": session_data[f"{question}_page"]
    }

@app.post("/feedback")
def feedback(request: FeedbackRequest):
    new_entry = {
        'question': request.question,
        'chosen_answer': request.chosen_answer
    }
    save_feedback(new_entry)
    session_data.pop(request.question, None)
    session_data.pop(f"{request.question}_page", None)
    return {"message": "Feedback guardado con éxito"}

@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de preguntas y respuestas"}

# Comando para ejecutar la aplicación: uvicorn main:app --reload
