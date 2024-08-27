
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
import streamlit as st

def get_response(user_query: str):


   # Definisci il prompt per l'analisi del sentiment
    template = "Analyze the sentiment of the following text and provide a sentiment score from -1 to 1, where -1 is very negative, 0 is neutral, and 1 is very positive:\n\n{text}"


    prompt_response = ChatPromptTemplate.from_template(template)
    llm = Ollama(model="seandearnaley/gemma-2-sentiment_analysis_with_reasoning:2b-q8_0", temperature = 0)
    full_chain = (
            prompt_response
            | llm
    )
    return full_chain.invoke({
        "text": user_query
    })

st.set_page_config(page_title= "Sentiment Analysis", page_icon=":speech_balloon:") #layout="wide"
st.title("Sentiment Analysis")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content = "Hello! I am a system capable of performing sentiment analysis. Please enter your review, and I will analyze the sentiment for you. System created by Daniele Pica and Chiara Saccone.")
    ]

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
        response = get_response(user_question)
        st.markdown(response)
    st.session_state.chat_history.append(AIMessage(content=response))
    #with st.spinner("Queryng database..."):


