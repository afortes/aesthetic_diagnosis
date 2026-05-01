import os
import yaml
import streamlit as st
import streamlit_authenticator as stauth
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI

load_dotenv()

# --- 1. CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Mi Buscador RAG", page_icon="🔍", layout="centered")

# --- 2. AUTENTICACIÓN ---
# En Cloud Run, credentials.yaml se monta como secreto en /run/secrets/CREDENTIALS_YAML
credentials_path = "/run/secrets/CREDENTIALS_YAML" if os.path.exists("/run/secrets/CREDENTIALS_YAML") else "credentials.yaml"
with open(credentials_path) as f:
    config = yaml.safe_load(f)

authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
)

authenticator.login()

if not st.session_state.get("authentication_status"):
    if st.session_state.get("authentication_status") is False:
        st.error("Usuario o contraseña incorrectos.")
    st.stop()

authenticator.logout("Cerrar sesión", "sidebar")
st.sidebar.write(f"Bienvenido, **{st.session_state['name']}**")

# --- 3. INICIALIZAR CLIENTES ---
st.title("Buscador Inteligente RAG 🔍")
st.markdown("Pregúntame cualquier cosa sobre los documentos, presentaciones o vídeos indexados.")

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(host=os.environ["PINECONE_INDEX_HOST"])
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# --- 3. GESTIÓN DEL HISTORIAL DE CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Dibujar los mensajes guardados en la pantalla
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. LÓGICA DE BÚSQUEDA Y RESPUESTA ---
# Esta caja de texto se queda fija en la parte inferior
user_query = st.chat_input("Escribe tu pregunta aquí (ej. Explícame qué es Dr. Baumann)...")

if user_query:
    # Mostrar la pregunta del usuario en la interfaz y guardarla en el historial
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Preparar el contenedor para la respuesta del asistente
    with st.chat_message("assistant"):
        with st.spinner("Buscando en la base de conocimiento..."):
            
            # 4.1 Búsqueda en Pinecone (Solicitando también los metadatos que creamos)
            results = index.search(
                namespace="example-namespace", 
                query={
                    "inputs": {"text": user_query}, 
                    "top_k": 4
                },
                fields=["category", "chunk_text", "source_file", "slide_number", "start_time"]
            )
            
            hits = results.get("result", {}).get("hits", [])
            
            if not hits:
                st.warning("No he encontrado información relevante en mis documentos para responder a esto.")
                st.stop()

            # 4.2 Ensamblar el contexto y formatear las fuentes
            context = ""
            fuentes_visuales = []
            
            for i, h in enumerate(hits):
                fields = h.get("fields", {})
                texto = fields.get("chunk_text", "").strip()
                categoria = fields.get("category", "general")
                archivo = fields.get("source_file", "Documento desconocido")
                
                # Añadimos el texto al contexto que leerá el LLM
                context += f"--- Fragmento {i+1} ---\n{texto}\n\n"
                
                # Formateamos la fuente visualmente para el usuario según el tipo de archivo
                if categoria == "presentacion":
                    diapo = int(fields.get("slide_number", 0))
                    fuentes_visuales.append(f"📄 **{archivo}** (Diapositiva {diapo})")
                elif categoria == "video":
                    inicio = float(fields.get("start_time", 0))
                    minutos = int(inicio // 60)
                    segundos = int(inicio % 60)
                    fuentes_visuales.append(f"🎬 **{archivo}** (Minuto {minutos}:{segundos:02d})")
                else:
                    fuentes_visuales.append(f"📝 **{archivo}**")

            # 4.3 Generar respuesta con OpenAI
            prompt = f"""Usa la siguiente información de contexto para responder la pregunta del usuario de forma clara y amable. 
            Si la respuesta no está en el contexto, indícalo claramente y no inventes información.

            Contexto:
            {context}

            Pregunta: {user_query}
            """

            # Usamos la sintaxis correcta de chat.completions y el modelo gpt-4o-mini (más rápido, barato y capaz)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Eres un asistente experto que ayuda a los usuarios a encontrar información en una base de datos documental."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            answer = response.choices[0].message.content
            
            # 4.4 Mostrar la respuesta y las fuentes
            st.markdown(answer)
            
            # Usamos un 'expander' (acordeón) para que las fuentes no saturen la pantalla
            with st.expander("📚 Ver fuentes consultadas"):
                # Usamos set() para eliminar fuentes duplicadas si dos chunks vienen del mismo sitio
                for fuente in set(fuentes_visuales):
                    st.markdown(f"- {fuente}")
            
            # Guardar la respuesta (sin las fuentes) en el historial
            st.session_state.messages.append({"role": "assistant", "content": answer})