import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Cargar el modelo y el tokenizador
@st.cache_resource
def cargar_modelo():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return tokenizer, model

tokenizer, model = cargar_modelo()

# Crear un área de texto para la entrada del usuario
texto_entrada = st.text_area("Introduce el texto que deseas analizar:", height=200)

if st.button("Analizar"):
    if texto_entrada:
        # Tokenizar y realizar la clasificación
        inputs = tokenizer(texto_entrada, return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs)
        predicciones = outputs.logits.softmax(dim=-1)
        
        # Obtener la etiqueta con mayor probabilidad
        etiquetas = ['negativo', 'neutral', 'positivo']
        sentimiento = etiquetas[predicciones.argmax().item()]
        
        # Mostrar el resultado
        st.subheader("Análisis de sentimiento:")
        st.write(f"El texto tiene un sentimiento: {sentimiento}")
    else:
        st.warning("Por favor, introduce un texto para analizar.")

st.markdown("---")
st.write("Esta aplicación utiliza FinBERT para analizar el sentimiento de textos.")
