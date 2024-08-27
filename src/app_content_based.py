from typing import List, Tuple

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
from streamlit_searchbox import st_searchbox
import ast
import re

# function with list of labels
def get_all_users(searchterm: str) -> List[Tuple[str, any]]:
    results = st.session_state.database.run(f"SELECT user_id FROM books_info WHERE user_id LIKE '{searchterm}%';")
    valori_estratti = []
    try:
      tuples_list = ast.literal_eval(results)
     # Estrai i valori da ciascuna tupla
      valori_estratti = [tupla[0] for tupla in tuples_list]
    except SyntaxError as e:
        print(e)
    return valori_estratti

#Metodo per ottenere le informazioni dei libri già letti sulla base del titolo (verrà usato iterativamente per ogni libro letto)
def get_book_info(searchterm: str) -> str:
    result = st.session_state.database.run(f"SELECT title,description,authors,categories,ratingsCount FROM books WHERE title = '{searchterm}';")
    return result

#Metodo per ottenere i titoli di tutti i libri già letti e recensiti dell'utente
def get_books_read(searchterm: str) -> [str]:
    result = st.session_state.database.run(f"SELECT book_info FROM books_info WHERE user_id = '{searchterm}';")
    titles = re.findall(r'title:(.*?), authors', result)
    return titles
def connect_to_database(user: str, password: str,host: str, port: str, database: str):
    # Setup database
    db_postgres = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_postgres)
def connect_to_chroma_database():
    # create a new Chroma vector store (index)
    persist_directory = 'db_persist'
    embeddings = OllamaEmbeddings(model='mxbai-embed-large', show_progress=True)
    recommend_db = Chroma(embedding_function=embeddings, persist_directory=persist_directory)
    return recommend_db

def find_similar_book(book_info:str, vectorstore:Chroma, top_k=2):
    # Trova gli utenti più simili
    similar_book = vectorstore.similarity_search_with_score(book_info, k=top_k)
    #similar_users = [doc for doc in similar_users if doc[0].metadata['user_id'] != user_id]
    return similar_book


def get_response(user_query: str, recommend_db: Chroma):

    template = '''
    You are a recommender system. Your task is to recommend books based on the context provided. Here’s how you should work:

1. **Input Query**: An initial query from the user.

2. **User book already read **: List of books read and review from the user selected

3. **Context**: A list of book similar to the books read by primary user. This context will be provided to you and is crucial for generating recommendations.

4. **Recommendation Task**:
   - **Analyze Book Information**: Look at the books that are similar to the books read and reviewed by user.
   - **Generate Recommendations**: Suggest books that are in the list and have not been read by the primary user. You can choose only the books in the context section.

**Instructions**:
- You should only use the information provided in the context to make recommendations. Do not use any external knowledge or assumptions.
- Provide a list of book recommendations that the primary user might find interesting, ensuring these books are not in the list of books already read by the primary user.
- The books you suggest must be in the context section.

User question: {question}
User books already read: {book_already_read} : These books must not appear in the list of recommended books
Context: {context}
The answer you give must contain: Title of the book, author of the book, category of the book and why it was recommended.
'''

    #Trovo i libri letti dall'utente selezionato
    books_read = get_books_read(selected_value)
    print("Libri letti dall'utente selezionato")
    print(books_read)
    #Setto una lista vuota che conterrà le info per ogni libro letto
    books_read_info = []
    #Per ogni titolo, mi prendo tutte le altre informazioni dei libri
    for book in books_read:
        books_read_info.append(get_book_info(book))
    print( "Per ogni titolo, mi prendo tutte le altre informazioni dei libri")
    print(books_read_info)
    #Setto una lista vuota di libri raccomandati
    books_recommended = []
    #Per ogni libro con le sue informazioni, genero embeddings e trovo gli embeddings più simili e li riporto
    for books in books_read_info:
        books_recommended.append(find_similar_book(books, recommend_db))
    print("Per ogni libro con le sue informazioni, trovo gli embeddings più simili e li riporto")
    print((books_recommended))


    prompt_response = ChatPromptTemplate.from_template(template)
    llm3 = Ollama(model="llama2:chat")
    full_chain = (
            RunnablePassthrough.assign(context = lambda x: books_recommended).assign(book_already_read = lambda x: books_read_info)
            | prompt_response
            | llm3
    )
    return full_chain.invoke({
        "question": user_query
    })
def get_response_2(user_query: str, recommend_db: Chroma):

    template = '''
 You are a recommender system. Your task is to recommend books based on the context provided. Here’s how you should work:

1. **Input Query**: An initial query from the user.

2. **User book info **: List of books read and review from the user selected

3. **Context**: A list of users similar to the primary user, along with the books they have read and reviewed. This context will be provided to you and is crucial for generating recommendations.

4. **Recommendation Task**:
   - **Analyze Book Information**: Look at the books that these similar users have read and reviewed.
   - **Generate Recommendations**: Suggest books that have been read by similar users but have not been read by the primary user. You can choose only the books in the context section.

**Instructions**:
- You should only use the information provided in the context to make recommendations. Do not use any external knowledge or assumptions.
- Provide a list of book recommendations that the primary user might find interesting, ensuring these books are not in the list of books already read by the primary user.
- The books you suggest must be in the context where are the users similar to the primary user.

User question: {question}
User books already read: {book_already_read} : These books must not appear in the list of recommended books
Context: {context}
The answer you give must contain: Title of the book, author of the book, category of the book and why it was recommended.
'''

    #query = "If i liked Economics in one lesson, What other books like it could you suggest to me?"
    book_info = get_book_info(selected_value) #qui ho il mio book info
    prompt_response = ChatPromptTemplate.from_template(template)
    llm3 = Ollama(model="llama2:chat", temperature = 0)
    print(find_similar_book(book_info, recommend_db))
    full_chain = (
            RunnablePassthrough.assign(context = lambda x: find_similar_book(book_info, recommend_db)).assign(book_already_read = lambda x: book_info)
            | prompt_response
            | llm3
    )
    return full_chain.invoke({
        "question": user_query
    })

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content = "Hello! I'm a RC assistant. Ask me anything about your database")
    ]

st.set_page_config(page_title= "Content-based", page_icon=":speech_balloon:") #layout="wide"
st.title("Content-Based Recommender System")


'''Content-Based Filtering: Come Funziona?
Analisi delle Caratteristiche: Ogni libro (o qualsiasi altro oggetto) viene descritto attraverso un insieme di caratteristiche o attributi, come il genere, l'argomento, l'autore, le parole chiave, ecc.

Creazione del Profilo Utente: Il sistema crea un profilo per ogni utente basato sui libri che ha letto e apprezzato. Questo profilo utente viene costruito combinando le caratteristiche dei libri che l'utente ha valutato positivamente.

Raccomandazioni: Il sistema confronta il profilo dell'utente con le caratteristiche dei libri disponibili e suggerisce quelli che sono più simili ai libri precedentemente apprezzati dall'utente.

Al posto di confrontare gli embeddings degli utenti con quelli dei libri, dall'utente estrapolo i libri letti e poi cerco dal db tutte le informazioni sui iìlibri e una volta trovate confronto gliembeddings per trovare i libri simili.'''



with st.sidebar:
    st.subheader("Settings")
    st.write("This is a simple chat application using Database. Connect to the database and start chatting.")
    st.text_input("Host", value="localhost", key="Host")
    st.text_input("Port", value="5432", key="Port")
    st.text_input("Database", value="sii", key="Database")
    st.text_input("User", value="postgres", key="User")
    st.text_input("Password", type="password", value="postgres", key="Password")
    if st.button("Start"):
        with st.spinner(text="Connecting to the database..."):
            db = connect_to_database(
                st.session_state["User"],
                st.session_state["Password"],
                st.session_state["Host"],
                st.session_state["Port"],
                st.session_state["Database"]
            )
            st.session_state.database = db
            st.success("Connected to the database.")

# pass search function to searchbox
selected_value = st_searchbox(
    get_all_users,
    key="searchbox",
)
chroma_db = None
chroma_setted = False

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

#user_question = st.chat_input(f"Type a message...{selected_value}")
# Messaggio precompilato
default_message = f"Can you suggest more books for user with user_id = {selected_value}?"

# Creazione del campo di input con un valore predefinito
user_question = st.text_area("Type a message...", value=default_message, height=20)

# Azione quando l'utente preme un pulsante o invia il messaggio
if st.button('Send') and selected_value != None:
    if not chroma_setted:
        chroma_db = connect_to_chroma_database()
        chroma_setted = True
    st.session_state.chat_history.append(HumanMessage(content=user_question))
    with st.chat_message("Human"):
        st.markdown(user_question)

    with (st.chat_message("AI")):
        response = get_response(user_question, chroma_db)
        st.markdown(response)
    st.session_state.chat_history.append(AIMessage(content=response))
else:
    st.write(f"Pre-filled message not modified: {user_question}")


