import pytest
from transformers import pipeline

# Configurar el modelo de preguntas y respuestas
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

@pytest.mark.parametrize("context, question, expected", [
    ("La inteligencia artificial es la simulación de procesos de inteligencia humana por parte de sistemas informáticos.",
     "¿Qué es la inteligencia artificial?",
     "La simulación de procesos de inteligencia humana"),
    
    ("Tesla, Inc. es una empresa fundada en 2003 especializada en vehículos eléctricos.",
     "¿Cuándo se fundó Tesla?",
     "2003"),
])
def test_qa_model(context, question, expected):
    answer = qa_pipeline(question=question, context=context)
    assert expected in answer["answer"], f"Esperado: {expected}, Obtenido: {answer['answer']}"

if __name__ == "__main__":
    pytest.main()
