# 베이스 이미지로 Python 3.10 사용
FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 패키지 목록을 requirements.txt에 작성해야 합니다.
COPY requirements.txt .

# 필요한 패키지 설치를 위한 시스템 패키지 설치
RUN apt-get update && \
    apt-get install -y llvm llvm-dev && \
    apt-get clean

# Pip 업그레이드 및 패키지 설치
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 필요한 파일들을 컨테이너에 복사
COPY . .

# Tesseract OCR 설치
RUN apt-get update && apt-get install -y tesseract-ocr && apt-get clean

# FFmpeg 설치
RUN apt-get update && apt-get install -y ffmpeg && apt-get clean

# 컨테이너 실행 시 기본 명령어 설정
CMD ["python", "AutoCut_v3 beta.py"]