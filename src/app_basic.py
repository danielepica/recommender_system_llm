from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_community.llms import Ollama
from dotenv import load_dotenv
import streamlit as st
from streamlit_option_menu import option_menu

def connect_to_database():
    # create a new Chroma vector store (index)
    persist_directory = 'db_persist'
    embeddings = OllamaEmbeddings(model='mxbai-embed-large', show_progress=True)
    recommend_db = Chroma(embedding_function=embeddings, persist_directory=persist_directory)
    return recommend_db

def get_response_2(user_query: str, recommend_db: Chroma):

    template = '''
    You are a books recommender system.
    Based on the user's query and the retrieved context, provide a comprehensive response by suggesting something relevant. 
    Respond only using the context provided here.
    User question: {question}
    Context: {context}'''

    #query = "If i liked Economics in one lesson, What other books like it could you suggest to me?"
    def get_context(user_query):
        docs = recommend_db.similarity_search(user_query, k=5)
        return docs

    prompt_response = ChatPromptTemplate.from_template(template)
    llm3 = Ollama(model="llama3")
    full_chain = (
            RunnablePassthrough.assign(context = get_context(user_query)) #mi da errore, da vedere meglio
            | prompt_response
            | llm3
    )
    return full_chain.invoke({
        "question": user_query
    })

def get_response(user_query:str, recommend_db: Chroma):
    llm = Ollama(model="llama3")
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=recommend_db.as_retriever(),
                                     return_source_documents=True, verbose=True)
    result = qa({"query": user_query})
    return result['result']


if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content = "Hello! I'm a Books Recommender System created by Daniele Pica and Chiara Saccone.")
    ]

st.set_page_config(page_title= "Recommender System", page_icon=":speech_balloon:") #layout="wide"
st.title("Recommender System")

# 1. as sidebar menu
with st.sidebar:
    selected = option_menu("Main Menu", ["Home", 'Settings'],
        icons=['house', 'gear'], menu_icon="cast", default_index=1)
    if st.button("Start connected to the db"):
        with st.spinner(text="Connecting to the vectorial database..."):
            db = connect_to_database()
            st.session_state.database = db
            st.success("Connected to the database.")

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

user_question = st.chat_input("Type a message...")
if user_question is not None and user_question.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_question))

    with st.chat_message("Human"):
        st.markdown(user_question)

    with (st.chat_message("AI")):
        response = get_response_2(user_question, st.session_state.database)
        st.markdown(response)
    st.session_state.chat_history.append(AIMessage(content=response))
    #with st.spinner("Queryng database..."):


