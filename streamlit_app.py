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
        # Tokenizar y generar el resumen
        inputs = tokenizer([texto_entrada], max_length=1024, return_tensors="pt", truncation=True)
        summary_ids = model.generate(inputs["input_ids"], num_beams=4, min_length=30, max_length=100, early_stopping=True)
        resumen = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # Mostrar el resumen
        st.subheader("Resumen del texto:")
        st.write(resumen)
    else:
        st.warning("Por favor, introduce un texto para analizar.")

st.markdown("---")
st.write("Esta aplicación utiliza el modelo BART de Facebook para generar resúmenes de texto.")
