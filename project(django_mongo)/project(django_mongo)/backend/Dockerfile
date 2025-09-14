# backend/Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 시스템 의존성 (빌드 도구 + psycopg2 등 필요 시 대비)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl netcat-traditional \
  && rm -rf /var/lib/apt/lists/*

# 라이브러리 설치
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# 소스 복사
COPY . .

# 헬스체크: gunicorn 8000 응답 확인
HEALTHCHECK --interval=30s --timeout=5s --retries=5 \
  CMD curl -fsS http://localhost:8000/health/ || exit 1

# 엔트리포인트: DB 대기 후 collectstatic/migrate → gunicorn
CMD ["bash", "-lc", "\
  until nc -z $MONGO_INITDB_HOST 27017; do echo '⏳ waiting for mongo...'; sleep 2; done; \
  python manage.py collectstatic --noinput || true; \
  python manage.py migrate || true; \
  gunicorn config.wsgi:application --bind 0.0.0.0:8000 --workers 3 --threads 2 --timeout 60 \
"]
