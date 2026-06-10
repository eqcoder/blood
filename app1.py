import os
import json
import pickle
import datetime
import urllib.request
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from openai import OpenAI
# LangChain 및 AI 관련 라이브러리
from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

app = FastAPI(title="올인원 융합형 헌혈 도우미 챗봇 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================================
# [설정] 환경 변수 및 API 키 정보 설정
# =========================================================================
DATA_DIR = "./data"
DB_DIR = "./chroma_db"
MODEL_PATH = "model.pkl"

# 💥 네이버 개발자 센터에서 발급받은 키를 여기에 입력하세요.
NAVER_CLIENT_ID = "YOUR_NAVER_CLIENT_ID"       
NAVER_CLIENT_SECRET = "YOUR_NAVER_CLIENT_SECRET" 

# 폴더 자동 생성 방어 코드
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# =========================================================================
# [1단계] 외부 연동: 네이버 검색어 트렌드 API 호출 함수
# =========================================================================
def get_naver_search_trend():
    """네이버 데이터랩 API를 호출하여 최근 한 달간 헌혈 관련 키워드 트렌드를 요약합니다."""
    url = "https://openapi.naver.com/v1/datalab/search"
    
    end_date = datetime.datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
    
    body = {
        "startDate": start_date,
        "endDate": end_date,
        "timeUnit": "week",
        "keywordGroups": [
            {"groupName": "헌혈참여", "keywords": ["헌혈", "헌혈의집", "헌혈조건"]},
            {"groupName": "헌혈보상", "keywords": ["헌혈사은품", "헌혈기념품", "봉사시간"]}
        ]
    }
    
    try:
        request = urllib.request.Request(url)
        request.add_header("X-Naver-Client-Id", NAVER_CLIENT_ID)
        request.add_header("X-Naver-Client-Secret", NAVER_CLIENT_SECRET)
        request.add_header("Content-Type", "application/json")
        
        response = urllib.request.urlopen(request, data=json.dumps(body).encode("utf-8"))
        if response.getcode() == 200:
            data = json.loads(response.read().decode("utf-8"))
            summary = []
            for group in data.get('results', []):
                group_name = group['title']
                last_ratio = group['data'][-1]['ratio'] if group['data'] else 0
                summary.append(f"- {group_name} 분야 최근 검색 트렌드 지수: {last_ratio}/100")
            return "\n".join(summary)
        return "네이버 API 응답이 비정상적입니다."
    except Exception as e:
        return f"네이버 트렌드 데이터를 불러올 수 없습니다. (이유: {str(e)})"

# =========================================================================
# [2단계] 머신러닝: 예측 모델 파일(.pkl) 로드 로직
# =========================================================================
if os.path.exists(MODEL_PATH):
    print("[🎉] 이미 학습된 머신러닝 모델 파일을 발견했습니다. 즉시 로드합니다.")
    with open(MODEL_PATH, "rb") as f:
        ml_model = pickle.load(f)
else:
    print("[-] [경고] 학습된 모델 파일(pkl)이 없습니다. 임시 더미 모델로 대체합니다.")
    print("[-] 상용 전 반드시 'train_model.py'를 먼저 실행해 모델을 생성하세요.")

# =========================================================================
# [3단계] RAG: 비용 절감을 위한 로컬 벡터 DB 로드 및 저장 로직
# =========================================================================
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
    print("[🎉] 이미 기존에 학습된 벡터 DB가 존재합니다. 로컬 파일을 불러옵니다.")
    vectorstore = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        collection_name="blood_donation_advanced_rag"
    )
else:
    print("[*] 기존 벡터 DB가 없습니다. 최초 1회 문서 임베딩(API 호출)을 시작합니다...")
    
    docs = []
    pdf_loader = PyPDFDirectoryLoader(DATA_DIR)
    txt_loader = DirectoryLoader(DATA_DIR, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    
    try: docs.extend(pdf_loader.load()) 
    except Exception as e: print(f"[-] PDF 로드 중 건너뜀: {e}")
        
    try: docs.extend(txt_loader.load()) 
    except Exception as e: print(f"[-] TXT 로드 중 건너뜀: {e}")
    
    if not docs:
        print("[!] 'data' 폴더가 비어 있어 임시 기본 방어 도큐먼트를 생성합니다.")
        from langchain_core.documents import Document
        docs = [Document(page_content="전혈헌혈은 만 16세부터 69세까지 가능합니다. 신분증이 필요합니다.")]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings,
        collection_name="blood_donation_advanced_rag",
        persist_directory=DB_DIR
    )
    print(f"[+] 최초 임베딩이 성공적으로 완료되어 '{DB_DIR}'에 저장되었습니다.")

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# =========================================================================
# [4단계] LLM 인프라 및 다방향 라우팅 프롬프트 체인 설정
# =========================================================================
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

# 질문의 의도를 분기 처리하는 라우터용 프롬프트
router_template = """사용자의 질문을 읽고, 다음 3가지 카테고리 중 가장 알맞은 하나를 선택하세요.
1. 'PREDICT': 오늘 또는 특정 날씨/날짜의 헌혈자 수 수치 예측 및 통계를 구하는 질문
2. 'RAG': 일반적인 헌혈 자격 조건, 준비물, 부작용, 예약 방법, 링크 안내 등 지식 기반 질문

오직 단어('PREDICT', 'RAG') 중 하나로만 답변하세요. 다른 설명은 절대 하지 마세요.

사용자 질문: {question}
답변:"""
router_prompt = ChatPromptTemplate.from_template(router_template)
router_chain = router_prompt | llm | StrOutputParser()

# 일반 RAG 정보 안내용 템플릿
rag_template = """당신은 친절하고 전문적인 '헌혈 도우미 챗봇'입니다. 
제공된 [헌혈 지식베이스]만을 바탕으로 사용자의 질문에 답변해 주세요.

[중요 지침]
1. 반드시 제공된 문맥(Context)에 적혀있는 내용에 기반하여 정확하게 답변하세요.
2. 만약 문맥(Context) 내에 사용자가 즉시 이동할 수 있는 링크 정보(예: [텍스트](URL))가 포함되어 있다면, 사용자가 클릭하여 이동할 수 있도록 답변의 가장 마지막 줄에 해당 마크다운 링크 형식을 그대로 유지하여 포함해 주세요.
3. 관련 근거가 없다면 지어내지 말고 정중히 모른다고 답변하세요.

[헌혈 지식베이스]
{context}

사용자 질문: {question}
답변:"""
rag_prompt = ChatPromptTemplate.from_template(rag_template)

def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

# =========================================================================
# [5단계] FastAPI 엔드포인트 제어부 (라우팅 스위치)
# =========================================================================
class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        user_msg = request.message
        
        # 1. 챗봇이 유저의 진짜 의도를 판별 (RAG vs PREDICT vs MARKETING)
        intent = router_chain.invoke({"question": user_msg}).strip().upper()
        print(f"[*] 분석된 질문 카테고리: {intent}")
        
        # -----------------------------------------------------------------
        # [Case A] 마케팅 아이디어 및 트렌드 분석 추천 요청
        # -----------------------------------------------------------------
        if "MARKETING" in intent:
            
            # 네이버 실시간 데이터 수집
            # trend_context = get_naver_search_trend()
            
            # marketing_prompt = f"""당신은 대한적십자사의 최고 '헌혈 홍보 마케팅 컨설턴트'입니다.
            # 사용자가 헌혈 활성화를 위한 아이디어 또는 마케팅 전략을 요청했습니다. 
            # 아래 제공되는 [실시간 네이버 검색어 트렌드 분석 데이터]를 반영하여, 창의적이고 실질적인 마케팅 대안을 3가지로 요약해 제안해 주세요.
            
            # [실시간 네이버 검색어 트렌드 분석 데이터 (최근 30일 요약)]
            # # {trend_context}
            # *(참고: 검색 트렌드 지수가 낮을수록 대중의 관심이 식었다는 뜻이며, '헌혈보상' 지수가 높으면 사은품에 민감하다는 뜻입니다.)*
            
            # 사용자 질문: {user_msg}
            # 답변 (전문적이고 실질적인 전략으로 작성):"""
            
            # response = llm.invoke(marketing_prompt).content
            return {"reply": 1}
            
        # -----------------------------------------------------------------
        # [Case B] 날씨 데이터 기반 오늘의 헌혈자수 예측 요청
        # -----------------------------------------------------------------
        elif "PREDICT" in intent:
            import requests
            from datetime import datetime

            AUTH_KEY = "1UQ9Stx3RiyEPUrcd9Yssw"

            today = datetime.now().strftime("%Y%m%d")

            url = (
                "https://apihub.kma.go.kr/api/typ01/url/kma_sfcdd3.php"
                f"?tm1={today}"
                f"&tm2={today}"
                "&stn=108"
                "&help=1"
                f"&authKey={AUTH_KEY}"
            )

            text = requests.get(url).text

            data_line = None

            for line in text.splitlines():
                if line.startswith(today):
                    data_line = line
                    break
            print(text)
            if data_line is None:
                print("데이터를 찾을 수 없습니다.")
            else:
                cols = data_line.split()
                print("평균기온:", cols[10])
                print("최고기온:", cols[11])
                print("최저기온:", cols[13])
                print("강수량:", cols[38])
            
            # 하드디스크에서 가져온 모델로 가볍게 연산만 수행 (fit 없음)
            input_df = pd.DataFrame([[(cols[11]+cols[13])/2, cols[13], cols[11], cols[38], '서울']], columns=['평균기온', '최저기온', '최고기온', '강수량', '지역'])
            predicted_count = int(ml_model.predict(input_df)[0])
            
            narrative_prompt = f"""사용자가 오늘의 예상 헌혈자 수를 물어보았습니다. 
            머신러닝 모델이 예측한 값과 오늘 날씨 정보를 바탕으로 사용자에게 친절하고 자연스럽게 안내 멘트를 작성해 주세요.
            
            [예측 데이터]
            - 오늘 날씨: 기온 {cols[10]}°C, 강수량 {cols[38]}mm
            - AI 예측 모델 결과: 오늘 예상 헌혈자 수 약 {predicted_count}명
            
            답변에는 날씨 상태와 함께 "머신러닝 분석 결과 오늘 약 {predicted_count}명의 헌혈자가 동참할 것으로 예상됩니다"라는 맥락을 반드시 포함해 주세요."""
            
            response = llm.invoke(narrative_prompt).content
            return {"reply": response}
            
        # -----------------------------------------------------------------
        # [Case C] 지식베이스 기반 일반 헌혈 문진 및 정보 상담 (RAG)
        # -----------------------------------------------------------------
        else:
            response = rag_chain.invoke(user_msg)
            return {"reply": response}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    # 휴대폰 등 외부 접속용 포트 개방
    uvicorn.run(app, host="0.0.0.0", port=8000)