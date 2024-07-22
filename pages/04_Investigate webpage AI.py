import streamlit as st
from langchain.document_loaders import AsyncChromiumLoader, SitemapLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


# import sys
# import asyncio
# from fake_useragent import UserAgent

llm = ChatOpenAI(temperature=0.1)

answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5.
    If the answer answers the user question the score should be high, else it should be low.
    Make sure to always include the answer's score even if it's 0.
    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!
    Question: {question}
"""
)

st.set_page_config(
    page_title="Investigate webpage AI",
    page_icon="🔍",
)
st.title("🔍 Investigate webpage AI")
st.markdown(
    """
Welcome!
            
Use this chatbot to ask questions about a webiste!!

Investigate webpage AI designed to analyze and extract information from webpages.

Please upload your file on the sidebar!
"""
)

# example - https://www.sydneydancecompany.com/class-sitemap.xml

# if "win32" in sys.platform:
#     # Windows specific event-loop policy & cmd
#     asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
#     cmds = [["C:/Windows/system32/HOSTNAME.EXE"]]
# else:
#     # Unix default event-loop policy & cmds
#     cmds = [
#         ["du", "-sh", "/Users/fredrik/Desktop"],
#         ["du", "-sh", "/Users/fredrik"],
#         ["du", "-sh", "/Users/fredrik/Pictures"],
#     ]
# Initialize a UserAgent object
# ua = UserAgent()


html2text_transfomer = Html2TextTransformer()


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm
    answers = []
    for doc in docs:
        result = answers_chain.invoke(
            {
                "question": question,
                "context": docs,
            }
        )
        answers.append(result.content)
    st.write(answers)


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return str(soup.get_text()).replace("\n", " ").replace("\xa0", " ")


@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )

    loader = SitemapLoader(
        url,
        # filter_urls=[
        #     r"^(.*\/blog\/).*",
        # ],
        parsing_function=parse_page,
    )
    # Set a realistic user agent
    # loader.headers = {"User-Agent": ua.random}
    # if load to fast, it will be blocked
    loader.requests_per_second = 1
    docs = loader.load_and_split(text_splitter=splitter)
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vector_store.as_retriever()


with st.sidebar:
    url = st.text_input(
        "Write down a URL",
        placeholder="https://example.com",
    )

if url:
    # asyn chrominum loader for dynamic web site /
    # loader = AsyncChromiumLoader([url])
    # docs = loader.load()
    # transformed = html2text_transfomer.transform_documents(docs)
    # st.write(docs)

    if ".xml" not in url:
        with st.sidebar:
            st.error("please write down a Sitemap URL.")

    else:
        retriever = load_website(url)
        # docs = retriever.invoke("what kind of classes are there?")
        # st.write(docs)

        chain = {
            "docs": retriever,
            "question": RunnablePassthrough(),
        } | RunnableLambda(get_answers)

        chain.invoke("what kind of classes are there?")
