import streamlit as st
from transformers import pipeline

# Configurar modelo
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Interfaz de usuario
st.title("Chatbot de Preguntas y Respuestas")
st.write("Haz una pregunta basada en el contexto proporcionado.")

# Entradas del usuario
context = st.text_area("Contexto", "Escribe el texto aquí...")
question = st.text_input("Pregunta", "¿Qué quieres saber?")

# Generar respuesta
if st.button("Responder"):
    if context and question:
        answer = qa_pipeline(question=question, context=context)
        st.write(f"**Respuesta**: {answer['answer']}")
    else:
        st.write("Por favor, proporciona un contexto y una pregunta.")
