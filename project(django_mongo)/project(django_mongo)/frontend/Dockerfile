# frontend/Dockerfile
# 1) Build
FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# 2) Serve with Nginx
FROM nginx:alpine
# SPA & API 프록시 설정 반영 (프로젝트의 frontend/nginx.conf 사용)
COPY nginx.conf /etc/nginx/conf.d/default.conf
# 정적 빌드 산출물 복사
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
HEALTHCHECK --interval=30s --timeout=5s --retries=5 \
  CMD wget -qO- http://localhost/ >/dev/null 2>&1 || exit 1
