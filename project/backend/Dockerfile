# 1. JDK 17 이미지를 기반으로 빌드 환경 구성
FROM openjdk:17-jdk-slim as builder
WORKDIR /app

# gradlew 및 gradle wrapper 복사 + 실행 권한 부여
COPY gradlew ./
COPY gradle ./gradle
RUN chmod +x gradlew

# 프로젝트 설정 및 소스 복사
COPY build.gradle settings.gradle ./
COPY src ./src

# 빌드 실행
RUN ./gradlew build -x test

# 2. 실제 실행 환경 구성
FROM openjdk:17-jdk-slim
WORKDIR /app
COPY --from=builder /app/build/libs/*.jar app.jar
ENTRYPOINT ["java", "-jar", "/app/app.jar"]