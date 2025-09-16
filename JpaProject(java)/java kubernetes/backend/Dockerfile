# 1. JDK 17 이미지를 기반으로 빌드 환경 구성
FROM openjdk:17-jdk-slim as builder
WORKDIR /app
COPY build.gradle settings.gradle ./
COPY gradlew ./
COPY src ./src
RUN ./gradlew build -x test

# 2. 실제 실행 환경 구성
FROM openjdk:17-jdk-slim
COPY --from=builder /app/build/libs/*.jar app.jar
ENTRYPOINT ["java", "-jar", "/app.jar"]

#1) 빌드 스테이지
#FROM eclipse-temurin:17-jdk AS builder
#WORKDIR /app
#
#Gradle 캐시 살리기 위해 설정/의존 파일 먼저 복사
#COPY gradlew ./
#COPY gradle ./gradle
#COPY build.gradle settings.gradle ./
#(있다면) gradle.properties도 함께
#COPY gradle.properties ./
#윈도우에서 생긴 CRLF 제거 + 실행 권한 부여
#RUN sed -i 's/\r$//' gradlew && chmod +x gradlew
#
#의존성만 먼저 내려받아 캐시
#RUN ./gradlew --no-daemon dependencies || true
#
#실제 소스 복사
#COPY src ./src
#
#빌드
#RUN ./gradlew --no-daemon clean build -x test
#
#실제 실행 환경 구성
#FROM openjdk:17-jdk-slim
#COPY --from=builder /app/build/libs/*.jar app.jar
#ENTRYPOINT ["java", "-jar", "/app.jar"]