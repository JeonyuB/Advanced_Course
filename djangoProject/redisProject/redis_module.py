import redis
import psycopg2
import pandas as pd
from sqlalchemy import create_engine

# connect = psycopg2.connect(host="localhost", database="postgres", user="postgres", password="0000")



if __name__ == "__main__":
    # Redis 연결 (db=15, 문자열 자동 디코딩)
    r = redis.Redis(host="localhost", port=6379, db=15, decode_responses=True)
    engine = create_engine('postgresql+psycopg2://postgres:0000@localhost:5432/postgres')

    # (선택) 초기화: 같은 스크립트를 여러 번 돌릴 경우 emp 리스트/기존 해시가 중복될 수 있으니 먼저 제거
    existing = r.lrange("emp", 0, -1)
    if existing:
        r.delete(*existing)
    r.delete("emp")

# 샘플 데이터 쓰기
    r.hset("employee:1", mapping={
        "name": "John", "age": 22, "job": "frontend", "language": "python", "pay": 400
    })
    r.hset("employee:2", mapping={
        "name": "Peter", "age": 22, "job": "frontend", "language": "python", "pay": 400
    })
    r.hset("employee:3", mapping={
        "name": "Sue", "age": 22, "job": "frontend", "language": "python", "pay": 400
    })
    r.hset("employee:4", mapping={
        "name": "Susan", "age": 22, "job": "frontend", "language": "python", "pay": 400
    })

    r.lpush("emp", "employee:1", "employee:2", "employee:3", "employee:4")
    # 리스트에 키 저장
    emp_list = r.lrange("emp", 0,-1)


    data = [{key: value for key, value in r.hgetall(emp).items()} for emp in emp_list]
    # print(data)
    # exit()
    df = pd.DataFrame(data)
    print(df)
    df.to_sql("employee", con=engine, if_exists="append", index=False)


