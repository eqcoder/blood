import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from flask import Flask, request, jsonify, send_file
from openai import OpenAI
import os
# LangChain 문서 로더 및 텍스트 스플리터
from langchain_community.document_loaders import TextLoader, PyPDFDirectoryLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

app = Flask(__name__)
DATA_DIR = "./data"
DB_DIR = "./chroma_db"
MODEL_PATH = "model.pkl"
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
# CORS 설정 (모바일 및 외부 접속 허용)

# =========================================================================
# [중요] 1. 다중 파일 및 다양한 포맷(.txt, .pdf) 문서 자동 로드 시스템
# =========================================================================
DATA_DIR = "./data"
DB_DIR = "./chroma_db"  # 💥 임베딩된 데이터를 영구 저장할 폴더 추가

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# =========================================================================
# [핵심] 비용 절감을 위한 로컬 벡터 DB 로드 및 저장 로직
# =========================================================================
if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
    # 1. 이미 저장된 데이터가 있다면: API 호출 없이 디스크에서 바로 로드 (비용 0원)
    print("[🎉] 이미 기존에 학습된 벡터 DB가 존재합니다. 로컬 파일을 불러옵니다.")
    vectorstore = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        collection_name="blood_donation_advanced_rag"
    )
else:
    # 2. 처음 실행하거나 DB 폴더가 비어있다면: 딱 한 번만 문서를 읽고 API를 호출하여 저장
    print("[*] 기존 벡터 DB가 없습니다. 최초 1회 문서 임베딩(API 호출)을 시작합니다...")
    
    # 문서 로드
    docs = []
    pdf_loader = PyPDFDirectoryLoader(DATA_DIR)
    txt_loader = DirectoryLoader(DATA_DIR, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    
    try:
        docs.extend(pdf_loader.load())
    except Exception as e:
        print(f"[-] PDF 로드 중 건너뜀: {e}")
        
    try:
        docs.extend(txt_loader.load())
    except Exception as e:
        print(f"[-] TXT 로드 중 건너뜀: {e}")
    
    if not docs:
        raise ValueError(f"'{DATA_DIR}' 폴더에 텍스트나 PDF 파일을 먼저 넣어주세요!")

    # 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    
    # 💥 핵심: Chroma에 'persist_directory'를 지정하여 하드디스크에 물리 파일로 저장
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings,
        collection_name="blood_donation_advanced_rag",
        persist_directory=DB_DIR  # 이 경로에 데이터가 영구 저장됩니다.
    )
    print(f"[+] 최초 임베딩이 성공적으로 완료되어 '{DB_DIR}'에 저장되었습니다.")

# 검색기(Retriever) 설정
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# =========================================================================
# 3. LLM 및 RAG 프롬프트 체인 설정
# =========================================================================
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1) # 정확도를 위해 낮은 온도로 설정

template = """당신은 친절하고 전문적인 '헌혈 도우미 챗봇'입니다. 
제공된 [헌혈 지식베이스]만을 바탕으로 사용자의 질문에 답변해 주세요.

[중요 지침]
1. 반드시 제공된 문맥(Context)에 적혀있는 내용에 기반하여 정확하게 답변하세요.
2. 만약 문맥(Context) 내에 사용자가 즉시 이동할 수 있는 링크 정보(예: [텍스트](URL))가 포함되어 있다면, 사용자가 클릭하여 이동할 수 있도록 답변의 가장 마지막 줄에 해당 마크다운 링크 형식을 그대로 유지하여 포함해 주세요.
3. 관련 근거가 없다면 지어내지 말고 정중히 모른다고 답변하세요.

[헌혈 지식베이스]
{context}

사용자 질문: {question}
답변:"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    # 각 문서 조각들 사이에 구분선을 주어 합칩니다.
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

# RAG 동작 체인 연결
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# =========================================================================
# 4. FastAPI 웹 서비스 라우터
# =========================================================================
class ChatRequest(BaseModel):
    message: str
@app.route("/")
def home():
    return send_file("index.html")
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json

    response = client.responses.create(
        model="gpt-5",
        input=data["message"]
    )

    return jsonify({
        "reply": response.output_text
    })
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        response = rag_chain.invoke(request.message)
        return {"reply": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    # 외부 기기(휴대폰 등)에서 접속할 수 있도록 0.0.0.0 으로 개방합니다.
    app.run(host="0.0.0.0", port=8080)