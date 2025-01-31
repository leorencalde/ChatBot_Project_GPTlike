import streamlit as st
from transformers import pipeline
from PIL import Image

# Configurar modelo
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Personalización del diseño
st.set_page_config(page_title="Chatbot IA", page_icon="🤖", layout="centered")

# Cargar imagen de fondo
st.image("https://source.unsplash.com/800x400/?technology,ai", use_column_width=True)

# Título y descripción
st.title("🤖 Chatbot de Preguntas y Respuestas")
st.markdown("""
### Haz una pregunta sobre el contexto proporcionado y obtén una respuesta inmediata. 🚀
""")

# Diseño de columnas para mejor distribución
col1, col2 = st.columns([2, 1])

with col1:
    context = st.text_area("📜 **Contexto**", "Escribe el texto aquí...", height=200)
with col2:
    st.info("💡 Consejo: Cuanto más claro sea el contexto, mejor será la respuesta del chatbot.")

question = st.text_input("❓ **Pregunta**", "¿Qué quieres saber?")

# Espacio para historial de preguntas
if "history" not in st.session_state:
    st.session_state.history = []

# Generar respuesta con efectos visuales
if st.button("💬 Responder"):
    if context and question:
        with st.spinner("🤖 Generando respuesta..."):
            answer = qa_pipeline(question=question, context=context)
            st.session_state.history.append((question, answer["answer"]))
            st.success(f"**Respuesta:** {answer['answer']}")
    else:
        st.warning("⚠️ Por favor, proporciona un contexto y una pregunta.")

# Mostrar historial de preguntas
if st.session_state.history:
    st.subheader("📜 Historial de Preguntas")
    for q, a in st.session_state.history[-5:]:  # Muestra las últimas 5 preguntas
        with st.expander(f"❓ {q}"):
            st.write(f"**Respuesta:** {a}")


