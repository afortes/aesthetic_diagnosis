import os
import yaml
from datetime import datetime, timedelta
import streamlit as st
import streamlit_authenticator as stauth
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI
from langfuse import observe, get_client, propagate_attributes

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

MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_MINUTES = 15

if "login_attempts" not in st.session_state:
    st.session_state.login_attempts = 0
if "lockout_until" not in st.session_state:
    st.session_state.lockout_until = None

if st.session_state.lockout_until:
    remaining = (st.session_state.lockout_until - datetime.now()).total_seconds()
    if remaining > 0:
        mins, secs = int(remaining // 60), int(remaining % 60)
        st.error(f"Demasiados intentos fallidos. Inténtalo de nuevo en {mins}m {secs}s.")
        st.stop()
    else:
        st.session_state.lockout_until = None
        st.session_state.login_attempts = 0

authenticator.login()

if not st.session_state.get("authentication_status"):
    if st.session_state.get("authentication_status") is False:
        st.session_state.login_attempts += 1
        left = MAX_LOGIN_ATTEMPTS - st.session_state.login_attempts
        if left <= 0:
            st.session_state.lockout_until = datetime.now() + timedelta(minutes=LOCKOUT_MINUTES)
            st.error(f"Demasiados intentos fallidos. Acceso bloqueado por {LOCKOUT_MINUTES} minutos.")
        else:
            st.error(f"Usuario o contraseña incorrectos. Intentos restantes: {left}.")
    st.stop()

authenticator.logout("Cerrar sesión", "sidebar")
st.sidebar.write(f"Bienvenido, **{st.session_state['name']}**")

# --- 3. INICIALIZAR CLIENTES ---
st.title("Buscador Inteligente RAG 🔍")
st.markdown("Pregúntame cualquier cosa sobre los documentos, presentaciones o vídeos indexados.")

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(host=os.environ["PINECONE_INDEX_HOST"])
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# --- FUNCIONES INSTRUMENTADAS CON LANGFUSE ---
@observe(name="pinecone-search")
def search_pinecone(query):
    results = index.search(
        namespace="example-namespace",
        query={"inputs": {"text": query}, "top_k": 4},
        fields=["category", "chunk_text", "source_file", "slide_number", "start_time"]
    )
    hits = results.get("result", {}).get("hits", [])
    get_client().update_current_span(output={"hit_count": len(hits)})
    return hits


@observe(name="openai-completion")
def generate_answer(messages):
    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    answer = response.choices[0].message.content
    get_client().update_current_generation(
        output=answer,
        usage_details={"input": response.usage.prompt_tokens, "output": response.usage.completion_tokens},
    )
    return answer


@observe(name="rag-query")
def rag_query(user_query):
    hits = search_pinecone(user_query)

    context = ""
    fuentes_visuales = []
    for i, h in enumerate(hits):
        fields = h.get("fields", {})
        texto = fields.get("chunk_text", "").strip()
        categoria = fields.get("category", "general")
        archivo = fields.get("source_file", "Documento desconocido")
        context += f"--- Fragmento {i+1} ---\n{texto}\n\n"
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

    prompt = f"""Usa la siguiente información de contexto para responder la pregunta del usuario de forma clara y amable.
            Si la respuesta no está en el contexto, indícalo claramente y no inventes información.

            Contexto:
            {context}

            Pregunta: {user_query}
            """
    messages = [
        {"role": "system", "content": "Eres un asistente experto que ayuda a los usuarios a encontrar información en una base de datos documental."},
        {"role": "user", "content": prompt}
    ]

    answer = generate_answer(messages)
    get_client().update_current_span(output=answer)
    return hits, fuentes_visuales, answer


# --- 4. GESTIÓN DEL HISTORIAL DE CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 5. LÓGICA DE BÚSQUEDA Y RESPUESTA ---
user_query = st.chat_input("Escribe tu pregunta aquí (ej. Explícame qué es Dr. Baumann)...")

if user_query:
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        with st.spinner("Buscando en la base de conocimiento..."):
            with propagate_attributes(
                user_id=st.session_state.get("name", "anonymous"),
                session_id=st.session_state.get("username", "unknown"),
            ):
                hits, fuentes_visuales, answer = rag_query(user_query)

            if not hits:
                st.warning("No he encontrado información relevante en mis documentos para responder a esto.")
                st.stop()

            st.markdown(answer)

            with st.expander("📚 Ver fuentes consultadas"):
                for fuente in set(fuentes_visuales):
                    st.markdown(f"- {fuente}")

            st.session_state.messages.append({"role": "assistant", "content": answer})
