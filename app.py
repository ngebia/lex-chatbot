import streamlit as st
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_groq import ChatGroq

st.set_page_config(page_title='Lex Fridman Podcast Chatbot', page_icon='🎙️', layout='wide')
st.title('🎙️ Lex Fridman Podcast Chatbot')
st.markdown('Ask anything from the **Lex Fridman Podcast** transcripts.')
st.divider()

@st.cache_resource
def load_chain():
    emb = SentenceTransformerEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2')
    vs = Chroma(
        collection_name='lex_fridman_podcasts',
        embedding_function=emb,
        persist_directory='./chroma_db')
    retriever = vs.as_retriever(search_kwargs={'k': 4})
    PROMPT = PromptTemplate(
        input_variables=['context', 'question'],
        template='''You are a knowledgeable assistant on the Lex Fridman Podcast.
Use the transcript excerpts below to answer accurately.
If the answer is not in the excerpts, say so honestly.

Context:
{context}

Question: {question}
Answer:''')
    llm = ChatGroq(
        model='llama3-8b-8192',
        temperature=0.2,
        api_key=st.secrets['GROQ_API_KEY'])
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT | llm | StrOutputParser()
    )
    return retriever, chain

retriever, chain = load_chain()

if 'messages' not in st.session_state:
    st.session_state.messages = []
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

if prompt := st.chat_input('Ask about any Lex Fridman episode...'):
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    with st.chat_message('user'): st.markdown(prompt)
    with st.chat_message('assistant'):
        with st.spinner('Searching transcripts...'):
            answer = chain.invoke(prompt)
            docs = retriever.invoke(prompt)
        st.markdown(answer)
        with st.expander('Source excerpts used'):
            for i, doc in enumerate(docs):
                ep = doc.metadata.get('episode_title', 'Unknown')
                guest = doc.metadata.get('guest', 'Unknown')
                st.markdown(f'**Source {i+1} — {ep} (guest: {guest})**')
                st.text(doc.page_content[:300] + '...')
                st.divider()
    st.session_state.messages.append({'role': 'assistant', 'content': answer})
