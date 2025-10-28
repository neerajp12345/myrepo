# RAG WebApp with Gemini and LangChain
# Compatible package versions - install with:
# pip install langchain==0.3.0 langchain-community==0.3.0 langchain-text-splitters==0.3.0 langchain-google-genai==2.0.0 chromadb==0.5.0 flask==3.0.3 python-dotenv==1.0.1 langchain-core==0.3.0

import os
from flask import Flask, render_template_string, request, jsonify
from dotenv import load_dotenv
from flask import Flask, render_template

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize Gemini model and embeddings
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY in .env file")

# Initialize embeddings and LLM
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7,
    convert_system_message_to_human=True
)

# Global vectorstore
vectorstore = None

# Custom prompt template for RAG
prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Answer: """

PROMPT = PromptTemplate(
    template=prompt_template, 
    input_variables=["context", "question"]
)


def process_documents(documents_text):
    """Process documents and create vector store"""
    global vectorstore
    
    # Create Document objects
    docs = [Document(page_content=documents_text)]
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    splits = text_splitter.split_documents(docs)
    
    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    return len(splits)


def get_qa_chain():
    """Create QA chain"""
    if vectorstore is None:
        return None
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    
    return qa_chain

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/upload', methods=['POST'])
def upload():
    try:
        data = request.json
        documents = data.get('documents', '')
        
        if not documents:
            return jsonify({'success': False, 'error': 'No documents provided'})
        
        chunks = process_documents(documents)
        
        return jsonify({
            'success': True,
            'chunks': chunks,
            'message': 'Documents processed successfully'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.json
        question = data.get('question', '')
        
        if not question:
            return jsonify({'success': False, 'error': 'No question provided'})
        
        if vectorstore is None:
            return jsonify({
                'success': False, 
                'error': 'Please upload documents first'
            })
        
        qa_chain = get_qa_chain()
        result = qa_chain({"query": question})
        
        return jsonify({
            'success': True,
            'answer': result['result'],
            'source_documents': len(result.get('source_documents', []))
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    print("Starting RAG WebApp with Gemini...")
    print("Make sure to set GOOGLE_API_KEY in your .env file")
    app.run(debug=True, port=5000)