import pandas as pd  # 데이터 프레임 처리용
import numpy as np  # 수치 연산용
import pickle  # 데이터 파일 로드용
import os
from sklearn.decomposition import TruncatedSVD, NMF  # 행렬분해용
from sqlalchemy.dialects.mssql.information_schema import columns
# from db_conn.postgres_db import conn_postgres_db  # 데이터베이스 연결용
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.sparse import lil_matrix
from tqdm import tqdm
import implicit
from threadpoolctl import threadpool_limits
# from lightfm import LightFM
# from lightfm.data import Dataset
import warnings
warnings.filterwarnings('ignore')

def extract_high_rating_data(minimum_rating=3.5):
    # pickle 파일에서 평점 데이터 로드
    with open("data/ratings.pkl", "rb") as f:
        df = pickle.load(f)
    userID = "user_id"
    movieID = "movie_id"
    users = df[userID].value_counts().index[:100] # 영화를 많은 본 사람의 랭킹 1000개
    movies = df[movieID].value_counts().index[:]


    data = df[(df[userID].isin(users)) & (df[movieID].isin(movies)) & (df['rating'] >= minimum_rating)]
    data.rename(columns={userID: "user_id", movieID: "movie_id"}, inplace=True)
    return data

def svd_predict_model(users, degree):
    """
    SVD(특이값 분해)를 사용한 협업 필터링 추천 시스템

    Args:
      users: 사용자-영화-평점 데이터프레임
      degree: SVD에서 사용할 차원 수 (잠재 요인 개수)

    Returns:
      예측된 평점이 담긴 데이터프레임 (user_id, movie_id, predicted_rating)
    """

    # ================================
    # 1단계: 피벗 테이블 생성
    # ================================
    # 사용자를 행(index), 영화를 열(columns)로 하는 평점 매트릭스 생성
    # fill_value=None으로 설정하여 평점이 없는 경우 NaN으로 처리
    pivot_rating = users.pivot_table(
        index="user_id", columns="movie_id", values="rating", fill_value=None)

    # ================================
    # 2단계: 빈 평점(Nan) 데이터 처리
    # ================================
    # 각 영화별 평균 평점 계산 (NaN 제외)
    random_mean = pivot_rating.mean(axis=0)

    # 빈 평점을 해당 영화의 평균 평점으로 채우기
    pivot_rating.fillna(random_mean, inplace=True)

    # ================================
    # 3단계: SVD 행렬 분해 준비
    # ================================
    # DataFrame을 numpy 배열로 변환 (SVD 알고리즘 입력용)
    matrix = pivot_rating.values

    # ================================
    # 4단계: SVD 행렬 분해 실행
    # ================================
    # TruncatedSVD: 큰 행렬을 효율적으로 특이값 분해하는 알고리즘
    # n_components=degree: 잠재 요인의 개수 (차원 축소)
    # random_state=42: 재현 가능한 결과를 위한 시드값
    svd = TruncatedSVD(n_components=degree, random_state=42)

    # fit_transform: 사용자 잠재 요인 행렬 생성
    # 각 사용자를 'degree'개의 잠재 요인으로 표현
    user_latent_matrix = svd.fit_transform(matrix)

    # components_: 영화 잠재 요인 행렬
    # 각 영화를 'degree'개의 잠재 요인으로 표현
    item_latent_matrix = svd.components_

    # ================================
    # 5단계: 예측 평점 계산
    # ================================
    # 행렬 곱셈으로 모든 사용자-영화 조합의 예측 평점 계산
    # user_latent_matrix @ item_latent_matrix = 예측 평점 행렬
    predicted_ratings = user_latent_matrix @ item_latent_matrix

    # ================================
    # 6단계: 결과를 DataFrame으로 변환
    # ================================
    # 원본 데이터의 고유한 사용자 ID와 영화 ID 추출
    index = users["user_id"].unique()  # 행 인덱스: 사용자 ID
    columns = users["movie_id"].unique()  # 열 인덱스: 영화 ID

    # 예측 평점을 DataFrame으로 변환 (피벗 테이블 형태)
    predicted_rating_df = pd.DataFrame(
        predicted_ratings, index=index, columns=columns)

    # ================================
    # 7단계: 피벗 해제 (Unpivot)
    # ================================
    # stack(): 피벗 테이블을 긴 형태(long format)로 변환
    # reset_index(): MultiIndex를 일반 컬럼으로 변환
    unpivot_predicted_rating_df = predicted_rating_df.stack().reset_index()

    # 컬럼명을 명확하게 설정
    unpivot_predicted_rating_df.columns = ["user_id", "movie_id", "predicted_rating"]

    return unpivot_predicted_rating_df

def performance_metrics(users, model_func, **model_kwargs):
    """
    주어진 모델 함수를 사용해 train/test split 후 성능을 평가합니다.

    Args:
        users (pd.DataFrame): 원본 사용자-영화-평점 데이터프레임
        model_func (callable): 모델 함수 (예: svd_predict_model, nmf_predict_model 등)
        **model_kwargs: 모델 함수에 전달할 파라미터 (예: degree=50)

    Returns:
        pd.DataFrame: test set에 대해 실제 평점과 예측 평점을 포함한 DataFrame
    """

    # 1. Train/Test Split
    train_data, test_data = train_test_split(users, test_size=0.2, random_state=42)

    # 2. 모델 실행 (train_data만 사용하여 학습)
    predicted_df = model_func(train_data, **model_kwargs)

    # 3. 예측값과 실제 test 데이터 병합
    comparison_df = pd.merge(test_data, predicted_df, on=['user_id', 'movie_id'], how='inner')

    # 4. 성능 지표 계산
    actual_ratings = comparison_df['rating']
    predicted_ratings = comparison_df['predicted_rating']

    rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
    mae = mean_absolute_error(actual_ratings, predicted_ratings)

    print(f"\n📊 {model_func.__name__} 성능 (params={model_kwargs})")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

    return comparison_df


def nmf_predict_model(users, degree):
    """
    NMF(비음수 행렬 분해)를 사용한 협업 필터링 추천 시스템

    Args:
      users: 사용자-영화-평점 데이터프레임
      degree: NMF에서 사용할 차원 수 (잠재 요인 개수)

    Returns:
      예측된 평점이 담긴 데이터프레임 (user_id, movie_id, predicted_rating)
    """

    # ================================
    # 1단계: 피벗 테이블 생성
    # ================================
    # 사용자를 행(index), 영화를 열(columns)로 하는 평점 매트릭스 생성
    # fill_value=None으로 설정하여 평점이 없는 경우 NaN으로 처리
    pivot_rating = users.pivot_table(
        index="user_id", columns="movie_id", values="rating", fill_value=None)

    # ================================
    # 2단계: 빈 평점(NaN) 데이터 처리
    # ================================
    # 각 영화별 평균 평점 계산 (NaN 제외)
    random_mean = pivot_rating.mean(axis=0)

    # 빈 평점을 해당 영화의 평균 평점으로 채우기
    pivot_rating.fillna(random_mean, inplace=True)

    # ================================
    # 3단계: NMF 행렬 분해 준비
    # ================================
    # DataFrame을 numpy 배열로 변환 (NMF 알고리즘 입력용)
    matrix = pivot_rating.values

    # NMF는 비음수(non-negative) 값만 허용하므로 음수가 있으면 0으로 처리
    # 평점 데이터는 일반적으로 양수이므로 대부분 문제없음
    matrix = np.maximum(matrix, 0)

    # ================================
    # 4단계: NMF 행렬 분해 실행
    # ================================
    # NMF: 행렬을 두 개의 비음수 행렬의 곱으로 분해하는 알고리즘
    # n_components=degree: 잠재 요인의 개수 (차원 축소)
    # init='random': 랜덤 초기화 방법
    # max_iter=500: 최대 반복 횟수
    # tol=1e-5: 수렴 허용 오차
    # random_state=42: 재현 가능한 결과를 위한 시드값
    nmf = NMF(n_components=degree, random_state=42, init='random', max_iter=500, tol=1e-5)

    # fit_transform: 사용자 잠재 요인 행렬 생성
    # 각 사용자를 'degree'개의 잠재 요인으로 표현
    P = nmf.fit_transform(matrix)

    # components_: 영화 잠재 요인 행렬
    # 각 영화를 'degree'개의 잠재 요인으로 표현
    Q = nmf.components_

    # ================================
    # 5단계: 예측 평점 계산
    # ================================
    # 행렬 곱셈으로 모든 사용자-영화 조합의 예측 평점 계산
    predicted_ratings = P @ Q

    # ================================
    # 6단계: 결과를 DataFrame으로 변환
    # ================================
    # 원본 데이터의 고유한 사용자 ID와 영화 ID 추출
    index = users["user_id"].unique()  # 행 인덱스: 사용자 ID
    columns = users["movie_id"].unique()  # 열 인덱스: 영화 ID

    # 예측 평점을 DataFrame으로 변환 (피벗 테이블 형태)
    predicted_rating_df = pd.DataFrame(
        predicted_ratings, index=index, columns=columns)

    # ================================
    # 7단계: 피벗 해제 (Unpivot)
    # ================================
    # stack(): 피벗 테이블을 긴 형태(long format)로 변환
    # reset_index(): MultiIndex를 일반 컬럼으로 변환
    unpivot_predicted_rating_df = predicted_rating_df.stack().reset_index()

    # 컬럼명을 명확하게 설정
    unpivot_predicted_rating_df.columns = ["user_id", "movie_id", "predicted_rating"]

    return unpivot_predicted_rating_df

def imf_predict_model(users, factors=10, minimum_num_ratings=4, epochs=50):
    """
    IMF(Implicit Matrix Factorization)를 사용한 협업 필터링 추천 시스템

    Args:
      users: 사용자-영화-평점 데이터프레임
      factors: ALS에서 사용할 잠재 요인 수
      minimum_num_ratings: 최소 평점 개수 (이보다 적은 상호작용을 가진 사용자/영화 제외)
      epochs: 학습 반복 횟수

    Returns:
      각 사용자별 상위 N개 추천 영화 리스트
    """

    # ================================
    # 1단계: 데이터 필터링
    # ================================
    # 최소 평점 개수 이상의 상호작용을 가진 사용자만 선택


    user_counts = users["user_id"].value_counts()
    valid_users = user_counts[user_counts >= minimum_num_ratings].index

    # 최소 평점 개수 이상의 상호작용을 가진 영화만 선택
    movie_counts = users["movie_id"].value_counts()
    valid_movies = movie_counts[movie_counts >= minimum_num_ratings].index

    # 필터링된 데이터만 사용
    filtered_users = users[
        (users["user_id"].isin(valid_users)) & (users["movie_id"].isin(valid_movies))]

    # ================================
    # 2단계: 인덱스 매핑 생성
    # ================================
    # 필터링된 데이터를 기반으로 인덱스 매핑 생성
    num_users = filtered_users["user_id"].nunique()
    num_movies = filtered_users["movie_id"].nunique()

    user_id2index = {
        user_id: i for i, user_id in enumerate(filtered_users["user_id"].unique())}
    movie_id2index = {
        movie_id: i for i, movie_id in enumerate(filtered_users["movie_id"].unique())}

    # ================================
    # 3단계: 희소 행렬 생성
    # ================================
    # 사용자-영화 상호작용 행렬 생성
    matrix = lil_matrix((num_users, num_movies))

    # 모든 평점을 1.0으로 변환 (상호작용 여부만 고려)
    for _, row in tqdm(filtered_users.iterrows(), total=len(filtered_users)):
        user_idx = user_id2index[row["user_id"]]
        movie_idx = movie_id2index[row["movie_id"]]
        matrix[user_idx, movie_idx] = 1.0

    # ================================
    # 4단계: CSR 형태로 변환
    # ================================
    # 희소 행렬을 CSR(Compressed Sparse Row) 형태로 변환 (연산 효율성)
    matrix_csr = matrix.tocsr()

    # ================================
    # 5단계: ALS 모델 학습
    # ================================
    # AlternatingLeastSquares: implicit feedback을 위한 행렬분해 알고리즘
    # factors: 잠재 요인의 개수 (차원 축소)
    # iterations: 학습 반복 횟수
    # calculate_training_loss: 학습 손실 계산 여부
    # random_state: 재현 가능한 결과를 위한 시드값
    model = implicit.als.AlternatingLeastSquares(
        factors=factors,
        iterations=epochs,
        calculate_training_loss=True,
        random_state=42
    )

    # ================================
    # 6단계: 모델 학습 실행
    # ================================
    # threadpool_limits: BLAS 스레드 수 제한 (메모리 안정성)
    with threadpool_limits(limits=4, user_api="blas"):
        model.fit(matrix_csr)

    # ================================
    # 7단계: 추천 결과 생성
    # ================================
    # 각 사용자별 상위 N개 영화 추천
    predicted_model = model.recommend_all(matrix_csr, N=10)

    # ================================
    # 8단계: DataFrame 형태로 변환
    # ================================
    # 추천 결과를 저장할 리스트
    recommendations = []

    # 인덱스를 원본 ID로 변환하기 위한 역매핑 생성
    index2user_id = {v: k for k, v in user_id2index.items()}
    index2movie_id = {v: k for k, v in movie_id2index.items()}

    # recommend_all 결과를 올바른 형태로 변환
    for user_idx in range(len(predicted_model)):
        original_user_id = index2user_id[user_idx]
        user_recommendations = predicted_model[user_idx]

        # 각 사용자의 추천 영화들을 처리
        for rank, movie_idx in enumerate(user_recommendations):
            original_movie_id = index2movie_id[movie_idx]

            # 추천 점수는 순위 기반으로 계산 후 0~5점 범위로 스케일링
            # 1위가 5점, 마지막 순위가 0점에 가깝게 설정
            normalized_score = 1.0 - (rank / len(user_recommendations))  # 0~1 범위
            predicted_score = normalized_score * 5.0  # 0~5점 범위로 변환

            recommendations.append({
                'user_id': original_user_id,
                'movie_id': original_movie_id,
                'predicted_rating': float(predicted_score)
            })
    print(recommendations)

    # DataFrame으로 변환
    predicted_df = pd.DataFrame(recommendations)

    return predicted_df

def bpr_predict_model(users, factors=10, minimum_num_ratings=4, epochs=50):
    """
    IMF(Implicit Matrix Factorization)를 사용한 협업 필터링 추천 시스템

    Args:
      users: 사용자-영화-평점 데이터프레임
      factors: ALS에서 사용할 잠재 요인 수
      minimum_num_ratings: 최소 평점 개수 (이보다 적은 상호작용을 가진 사용자/영화 제외)
      epochs: 학습 반복 횟수

    Returns:
      각 사용자별 상위 N개 추천 영화 리스트
    """

    # ================================
    # 1단계: 데이터 필터링
    # ================================
    # 최소 평점 개수 이상의 상호작용을 가진 사용자만 선택
    user_counts = users["user_id"].value_counts()
    valid_users = user_counts[user_counts >= minimum_num_ratings].index

    # 최소 평점 개수 이상의 상호작용을 가진 영화만 선택
    movie_counts = users["movie_id"].value_counts()
    valid_movies = movie_counts[movie_counts >= minimum_num_ratings].index

    # 필터링된 데이터만 사용
    filtered_users = users[
        (users["user_id"].isin(valid_users)) & (users["movie_id"].isin(valid_movies))]

    # ================================
    # 2단계: 인덱스 매핑 생성
    # ================================
    # 필터링된 데이터를 기반으로 인덱스 매핑 생성
    num_users = filtered_users["user_id"].nunique()
    num_movies = filtered_users["movie_id"].nunique()

    user_id2index = {
        user_id: i for i, user_id in enumerate(filtered_users["user_id"].unique())}
    movie_id2index = {
        movie_id: i for i, movie_id in enumerate(filtered_users["movie_id"].unique())}

    # ================================
    # 3단계: 희소 행렬 생성
    # ================================
    # 사용자-영화 상호작용 행렬 생성
    matrix = lil_matrix((num_users, num_movies))

    # 모든 평점을 1.0으로 변환 (상호작용 여부만 고려)
    for _, row in tqdm(filtered_users.iterrows(), total=len(filtered_users)):
        user_idx = user_id2index[row["user_id"]]
        movie_idx = movie_id2index[row["movie_id"]]
        matrix[user_idx, movie_idx] = 1.0

    # ================================
    # 4단계: CSR 형태로 변환
    # ================================
    # 희소 행렬을 CSR(Compressed Sparse Row) 형태로 변환 (연산 효율성)
    matrix_csr = matrix.tocsr()

    # ================================
    # 5단계: ALS 모델 학습
    # ================================
    # AlternatingLeastSquares: implicit feedback을 위한 행렬분해 알고리즘
    # factors: 잠재 요인의 개수 (차원 축소)
    # iterations: 학습 반복 횟수
    # calculate_training_loss: 학습 손실 계산 여부
    # random_state: 재현 가능한 결과를 위한 시드값
    model = implicit.bpr.BayesianPersonalizedRanking(
        factors=factors,
        learning_rate=0.01,
        regularization=0.01,
        iterations=100,
        random_state=42
    )

    # ================================
    # 6단계: 모델 학습 실행
    # ================================
    # threadpool_limits: BLAS 스레드 수 제한 (메모리 안정성)
    with threadpool_limits(limits=4, user_api="blas"):
        model.fit(matrix_csr)

    # ================================
    # 7단계: 추천 결과 생성
    # ================================
    # 각 사용자별 상위 N개 영화 추천
    predicted_model = model.recommend_all(matrix_csr, N=10)

    # ================================
    # 8단계: DataFrame 형태로 변환
    # ================================
    # 추천 결과를 저장할 리스트
    recommendations = []

    # 인덱스를 원본 ID로 변환하기 위한 역매핑 생성
    index2user_id = {v: k for k, v in user_id2index.items()}
    index2movie_id = {v: k for k, v in movie_id2index.items()}

    # recommend_all 결과를 올바른 형태로 변환
    for user_idx in range(len(predicted_model)):
        original_user_id = index2user_id[user_idx]
        user_recommendations = predicted_model[user_idx]

        # 각 사용자의 추천 영화들을 처리
        for rank, movie_idx in enumerate(user_recommendations):
            original_movie_id = index2movie_id[movie_idx]

            # 추천 점수는 순위 기반으로 계산 후 0~5점 범위로 스케일링
            # 1위가 5점, 마지막 순위가 0점에 가깝게 설정
            normalized_score = 1.0 - (rank / len(user_recommendations))  # 0~1 범위
            predicted_score = normalized_score * 5.0  # 0~5점 범위로 변환

            recommendations.append({
                'user_id': original_user_id,
                'movie_id': original_movie_id,
                'predicted_rating': float(predicted_score)
            })
    print(recommendations)

    # DataFrame으로 변환
    predicted_df = pd.DataFrame(recommendations)

    return predicted_df

# def lightfm_bpr_predict_model(users, factors=2, minimum_num_ratings=4, epochs=50):
#     """
#     LightFM과 BPR Loss를 사용한 협업 필터링 추천 시스템
#
#     Args:
#       users (pd.DataFrame): 사용자-영화-평점 데이터프레임
#       factors (int): 잠재 요인(Latent Factor)의 수
#       minimum_num_ratings (int): 필터링을 위한 최소 평점 개수
#       epochs (int): 학습 반복 횟수
#
#     Returns:
#       pd.DataFrame: 각 사용자-영화 쌍에 대한 예측 점수 데이터프레임
#     """
#
#     # ================================
#     # 1단계: 데이터 필터링 (원본 코드와 동일)
#     # ================================
#     user_counts = users["user_id"].value_counts()
#     valid_users = user_counts[user_counts >= minimum_num_ratings].index
#
#     movie_counts = users["movie_id"].value_counts()
#     valid_movies = movie_counts[movie_counts >= minimum_num_ratings].index
#
#     filtered_data = users[
#         (users["user_id"].isin(valid_users)) & (users["movie_id"].isin(valid_movies))
#         ].copy()
#
#
#     # print(filtered_data)
#     # exit()
#
#     # 필터링된 데이터의 고유 사용자 및 영화 목록
#     unique_user_ids = sorted(filtered_data['user_id'].unique())
#     unique_movie_ids = sorted(filtered_data['movie_id'].unique())
#
#     # ================================
#     # 2단계: LightFM Dataset 생성 및 매핑
#     # ================================
#     # Dataset 객체를 생성하고 사용자/아이템 ID를 내부 인덱스에 매핑합니다.
#     dataset = Dataset()
#     dataset.fit(users=unique_user_ids, items=unique_movie_ids)
#
#     # 상호작용 행렬을 구축합니다. BPR은 상호작용 유무가 중요하므로 가중치는 1로 설정합니다.
#     (interactions, weights) = dataset.build_interactions(
#         (row['user_id'], row['movie_id'])
#         for index, row in filtered_data.iterrows()
#     )
#     # print(interactions)
#     # exit()
#     # ================================
#     # 3단계: BPR 모델 학습
#     # ================================
#     # loss='bpr': Bayesian Personalized Ranking 손실 함수를 사용합니다.
#     model = LightFM(
#         no_components=factors,
#         loss='bpr',
#         learning_rate=0.01,
#         random_state=42
#     )
#
#     # 모델 학습 실행
#     model.fit(
#         interactions,
#         epochs=epochs,
#         num_threads=os.cpu_count()  # 사용 가능한 CPU 코어 수에 맞게 조절
#     )
#
#
#     # ================================
#     # 4단계: 모든 사용자-영화 쌍에 대한 예측 점수 생성 (효율성 개선)
#     # ================================
#     # 예측 결과를 저장할 DataFrame 리스트
#     predictions_list = []
#
#     # Dataset에서 생성된 내부 매핑을 가져옵니다.
#     user_id_map, _, _, _ = dataset.mapping()
#
#     # tqdm을 사용하여 진행 상황 표시
#     for user_id in tqdm(unique_user_ids, desc="Predicting scores"):
#         # LightFM 모델은 내부 인덱스를 사용합니다.
#         user_idx = user_id_map[user_id]
#
#         # 해당 사용자에 대해 모든 영화의 점수를 예측합니다.
#         item_indices = np.arange(len(unique_movie_ids))
#         scores = model.predict(user_idx, item_indices)
#
#         # [수정된 부분]
#         # 예측 결과를 매번 append하는 대신, 사용자별로 DataFrame을 만들어 리스트에 저장합니다.
#         # 이 방식이 훨씬 빠릅니다.
#         df_temp = pd.DataFrame({
#             'user_id': user_id,
#             'movie_id': unique_movie_ids,
#             'score': scores
#         })
#         predictions_list.append(df_temp)
#
#     # [수정된 부분]
#     # 리스트에 저장된 모든 DataFrame을 한 번에 효율적으로 결합합니다.
#     predicted_df = pd.concat(predictions_list, ignore_index=True)
#
#     return predicted_df

if __name__ == "__main__":
    users = extract_high_rating_data()

    # ✅ SVD 평가
    svd_results = performance_metrics(users, svd_predict_model, degree=50)
    print(svd_results.head())

    # ✅ NMF 평가
    nmf_results = performance_metrics(users, nmf_predict_model, degree=20)
    print(nmf_results.head())

    # ✅ IMF (ALS) 평가
    imf_results = performance_metrics(users, imf_predict_model, factors=20, epochs=30)
    print(imf_results.head())

    # ✅ BPR 평가
    bpr_results = performance_metrics(users, bpr_predict_model, factors=20, epochs=30)
    print(bpr_results.head())

    exit()
    # print(users.shape)
    # exit()
    # degree=10: 10개의 잠재 요인으로 차원 축소
    # 결과: 모든 사용자-영화 조합의 예측 평점
    # users_df = svd_predict_model(users, 10)
    # conn_postgres_db(users_df, "kogo", "1111", "mydb", "svd_model")
    # print("svd success")
    # users_df = nmf_predict_model(users, 10)
    # conn_postgres_db(users_df, "kogo", "1111", "mydb", "nmf_model")
    # print("nmf success")
    # users_df = imf_predict_model(users)
    # conn_postgres_db(users_df, "kogo", "1111", "mydb", "imf_model")
    # print("imf success")
    # users_df = bpr_predict_model(users)
    # conn_postgres_db(users_df, "kogo", "1111", "mydb", "bpr_model")
    # print("bpr success")
    # users_df = lightfm_bpr_predict_model(users)
    # print(users_df)

    # print(users_df[users_df["user_id"]==143])

    # 전체 DataFrame 출력 설정
    # pd.set_option('display.max_rows', None)  # 모든 행 출력
    # pd.set_option('display.max_columns', None)  # 모든 열 출력
    # pd.set_option('display.width', None)  # 너비 제한 없음
    # pd.set_option('display.max_colwidth', None)  # 열 너비 제한 없음
    #
    # print(users_df)

    # ================================
    # 5단계: 데이터베이스에 저장
    # ================================
    # conn_postgres_db(users_df, "kogo", "1111", "mydb", "nmf_model")