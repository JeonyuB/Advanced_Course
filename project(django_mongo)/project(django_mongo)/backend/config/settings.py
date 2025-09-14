from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent

# =========================
# 기본 보안/디버그
# =========================
SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", "dev-only")
DEBUG = os.getenv("DJANGO_DEBUG", "False").lower() == "true"

def _split_env(name: str):
    v = os.getenv(name, "").strip()
    if not v:
        return []
    # "*" 단독 허용 처리
    if v == "*":
        return ["*"]
    return [x.strip() for x in v.split(",") if x.strip()]

ALLOWED_HOSTS = _split_env("DJANGO_ALLOWED_HOSTS") or ["localhost", "127.0.0.1"]

# 리버스 프록시(nginx) 뒤에서 HTTPS 판단
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")

# 운영 보안(필요 시 활성화)
if not DEBUG:
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
    # SECURE_HSTS_SECONDS = 31536000
    # SECURE_HSTS_INCLUDE_SUBDOMAINS = True
    # SECURE_HSTS_PRELOAD = True
    # SECURE_REFERRER_POLICY = "same-origin"

# =========================
# 앱 / 미들웨어
# =========================
INSTALLED_APPS = [
    "django.contrib.admin","django.contrib.auth","django.contrib.contenttypes",
    "django.contrib.sessions","django.contrib.messages","django.contrib.staticfiles",
    "app",
    "rest_framework",
    "corsheaders",
]

MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",  # CORS는 최상단 권장
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [{
    "BACKEND": "django.template.backends.django.DjangoTemplates",
    "DIRS": [],
    "APP_DIRS": True,
    "OPTIONS": {
        "context_processors": [
            "django.template.context_processors.debug",
            "django.template.context_processors.request",
            "django.contrib.auth.context_processors.auth",
            "django.contrib.messages.context_processors.messages",
        ],
    },
}]

WSGI_APPLICATION = "config.wsgi.application"

# =========================
# 데이터베이스 (Djongo / MongoDB)
# =========================
# 환경변수 우선: MONGO_URL=mongodb://user:pass@host:27017/?authSource=admin
MONGO_URL = os.getenv("MONGO_URL", "").strip()
MONGO_DB  = os.getenv("MONGO_INITDB_DATABASE", "mydb")
MONGO_HOST = os.getenv("MONGO_INITDB_HOST", "db")
MONGO_PORT = os.getenv("MONGO_INITDB_PORT", "27017")
MONGO_USER = os.getenv("MONGO_INITDB_ROOT_USERNAME", "")
MONGO_PASS = os.getenv("MONGO_INITDB_ROOT_PASSWORD", "")

if not MONGO_URL:
    MONGO_URL = f"mongodb://{MONGO_USER}:{MONGO_PASS}@{MONGO_HOST}:{MONGO_PORT}/?authSource=admin"

DATABASES = {
    "default": {
        "ENGINE": "djongo",
        "NAME": MONGO_DB,
        "CLIENT": {
            "host": MONGO_URL,   # pymongo URI 형태 권장
            "tls": False,
        },
    }
}

# =========================
# 인증/국제화
# =========================
AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "ko-kr"
TIME_ZONE = "Asia/Seoul"
USE_I18N = True
USE_TZ = False   # (로컬 시간 저장이 필요하면 False 유지)

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# =========================
# 정적 파일
# =========================
STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"   # 배포: collectstatic 대상
# 개발 편의를 위해 앱 내부 static만 쓰면 STATICFILES_DIRS 불필요

# =========================
# CORS/CSRF
# =========================
CORS_ALLOWED_ORIGINS = _split_env("CORS_ALLOWED_ORIGINS")
# 프론트에서 포트/도메인이 유동적이면 정규식 사용 가능(예시)
# CORS_ALLOWED_ORIGIN_REGEXES = [r"^http://(localhost|127\.0\.0\.1)(:\d+)?$"]
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOW_HEADERS = ["*"]
CORS_ALLOW_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]

CSRF_TRUSTED_ORIGINS = _split_env("CSRF_TRUSTED_ORIGINS")

# =========================
# DRF
# =========================
REST_FRAMEWORK = {
    "DEFAULT_RENDERER_CLASSES": ["rest_framework.renderers.JSONRenderer"],
    "DEFAULT_PARSER_CLASSES": ["rest_framework.parsers.JSONParser"],
}

# =========================
# 로깅(운영 트러블슈팅)
# =========================
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {"console": {"class": "logging.StreamHandler"}},
    "root": {"handlers": ["console"], "level": "INFO"},
}
