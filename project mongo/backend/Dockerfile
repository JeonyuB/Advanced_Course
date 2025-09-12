FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# myproject.wsgi는 실제 프로젝트 이름에 맞게 수정
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "myproject.wsgi:application"]