# Langchain-RAG

루비페이퍼 "RAG 시스템을 위한 랭체인 완전정복" 도서의 실습 코드 및 자료입니다.

## 프로젝트 구조

- `3장`: Models.ipynb, Output_Parser.ipynb, Prompt_Template.ipynb
- `4장`: PDF_Document_Loaders.ipynb, Other_Document_Loaders.ipynb, Text_Splitters.ipynb
- `5장`: Text Embedding.ipynb, Vector Stores.ipynb, Retriever.ipynb, Basic RAG.ipynb, LCEL.ipynb
- `6장`: streamlit_chat.py, streamlit_rag_local.py, streamlit_rag_memory.py, streamlit_rag_upload.py, Tool&Agent.ipynb
- `data`: 실습 코드 내 예제 파일
- `requirements.txt`: 프로젝트 의존성 파일

## 시작하기
**1. 가상환경을 만듭니다.**
   ```bash
   conda create -n [가상환경 이름] python=3.12
   ```
**2. 저장소를 클론합니다:**
   ```bash
   git clone https://github.com/Kane0002/Langchain-RAG.git
   ```

**3. 필요한 패키지를 설치합니다:**
   ```bash
   pip install -r requirements.txt
   ```
- 이 코드를 실행하면 각 실습 장의 필수 라이브러리를 설치하지 않아도 됩니다.
- 실습 파일마다 필수 라이브러리를 표시한 이유는, 해당 섹션만 실습을 진행하고자 할 때를 위함입니다.
**4. Jupyter Notebook을 실행하여 각 장의 내용을 확인하고 실습할 수 있습니다.**

## 주의사항
**1. Window OS 환경에서 오픈소스 LLM 실습 진행을 위한 torch 설치**
- 5장의 Basic RAG.ipynb에는 오픈소스LLM 기반 RAG 실습 파일이 포함되어 있습니다.
- 해당 실습 진행을 위해서는 PC 내 GPU CUDA 환경에 알맞는 torch 설치가 필요합니다.
- 현재 저자의 GPU RTX 4060의 경우 다음과 같은 설치 과정을 거쳤습니다.
  ```
  pip uninstall torch
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
  ```
**2. langchain-chroma 또는 chromadb 설치 과정 중 Window C++ Build Tools 에러**
- Chroma의 경우 설치 과정 상 Window C++ Build Tools를 설치하라는 에러가 발생할 수 있습니다.
- 이 때, 아래 링크에 접속하여 Build tools를 설치해야 합니다.
- https://visualstudio.microsoft.com/ko/downloads/?q=build+tools
- 위 링크를 통해 다운로드한 Visual Studio setup.exe 파일을 실행한 후, C++ Build Tools를 체크하고 설치를 진행하면 됩니다.

**3. ChromaDB 포함 코드 실행 시 커널 죽는 문제**
- ChromaDB는 타 패키지와의 의존성 충돌이 생기기 쉽기 때문에 가상환경 속에서 실습을 진행하시는 것을 추천합니다.
- 만약 기존 환경 내에서 ChromaDB 실습을 진행할 경우, 가상환경을 새로 구축하고 requirements.txt를 통해 한꺼번에 실습에 필요한 패키지들을 다운로드하세요.
- 모든 해결 방법이 작동하지 않는 경우, Colab에서의 실습 또는 FAISS로 VectorDB를 교체하시는 것을 추천합니다.
