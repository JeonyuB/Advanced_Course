from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import aioredis
import logging
from config.settings import settings

app = FastAPI(debug=settings.DEBUG)

# Redis 클라이언트 연결 (비동기)
redis = aioredis.from_url(settings.REDIS_URL, decode_responses=True)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ALLOWED_ORIGINS or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)

@app.get("/")
async def root():
    # Redis 예시: 키 "test" 값 가져오기
    value = await redis.get("test")
    return {"message": "Hello FastAPI with Redis!", "redis_test_value": value}
