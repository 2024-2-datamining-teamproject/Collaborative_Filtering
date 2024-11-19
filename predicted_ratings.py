import pandas as pd
import numpy as np

# 하이퍼파라미터 설정
latent_dim = 30  # 잠재 벡터의 차원
learning_rate = 0.005  # 학습률
lambda_reg = 0.1  # 정규화 파라미터
epochs = 30  # 학습 횟수

# 데이터 로드
ratings = pd.read_csv('ratings.csv')  # 'ratings.csv'는 userId, movieId, rating 컬럼을 포함
movies = pd.read_csv('movies.csv')  # 'movies.csv'는 movieId, title 컬럼을 포함

# 전체 movieId 가져오기 (movies.csv 기준)
all_movie_ids = movies['movieId'].unique()

# 사용자 및 영화 ID 매핑
user_ids = ratings['userId'].unique()
user_id_map = {id_: idx for idx, id_ in enumerate(user_ids)}
movie_id_map = {id_: idx for idx, id_ in enumerate(all_movie_ids)}  # 모든 movieId 포함

# `ratings`에서 사용자 및 영화 인덱스 추가
ratings['user_idx'] = ratings['userId'].map(user_id_map)
ratings['movie_idx'] = ratings['movieId'].map(movie_id_map)

# 사용자 및 영화의 개수
num_users = len(user_ids)
num_movies = len(all_movie_ids)

# 사용자와 영화 잠재 벡터 초기화
# P = np.random.normal(scale=1.0 / latent_dim, size=(num_users, latent_dim))  # 사용자 잠재 벡터
# Q = np.random.normal(scale=1.0 / latent_dim, size=(num_movies, latent_dim))  # 영화 잠재 벡터
P = np.random.uniform(-np.sqrt(6 / latent_dim), np.sqrt(6 / latent_dim), size=(num_users, latent_dim))
Q = np.random.uniform(-(np.sqrt(6 / latent_dim)), np.sqrt(6 / latent_dim), size=(num_movies, latent_dim))

# 평점 데이터 정규화 (최소-최대 정규화)
RATING_MIN = ratings['rating'].min()
RATING_MAX = ratings['rating'].max()
ratings['rating_normalized'] = (ratings['rating'] - RATING_MIN) / (RATING_MAX - RATING_MIN)

# SGD 학습
for epoch in range(epochs):
    total_loss = 0
    for _, row in ratings.iterrows():
        user_idx = int(row['user_idx'])
        movie_idx = int(row['movie_idx'])
        rating = row['rating_normalized']  # 정규화된 평점 사용

        # 예측 평점
        pred_rating = np.dot(P[user_idx], Q[movie_idx])

        # 오차 계산
        error = rating - pred_rating

        # 사용자와 영화 잠재 벡터 업데이트
        P[user_idx] += learning_rate * (error * Q[movie_idx] - lambda_reg * P[user_idx])
        Q[movie_idx] += learning_rate * (error * P[user_idx] - lambda_reg * Q[movie_idx])

        # 손실 계산 (옵션)
        total_loss += error**2 + lambda_reg * (np.linalg.norm(P[user_idx])**2 + np.linalg.norm(Q[movie_idx])**2)
    
    # Epoch 결과 출력
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

# 모든 사용자-영화 조합에 대해 예상 평점 계산
print("Calculating all predicted ratings...")
predicted_ratings = np.dot(P, Q.T)  # 사용자-영화 예상 평점 매트릭스

# 예측 평점 복원
predicted_ratings = predicted_ratings * (RATING_MAX - RATING_MIN) + RATING_MIN

# 예상 평점을 DataFrame으로 변환
# `movies.csv` 기준으로 모든 movieId 포함
predicted_ratings_df = pd.DataFrame(predicted_ratings, index=user_ids, columns=all_movie_ids)

# 예상 평점 범위를 [0, 5]로 제한
predicted_ratings_df = predicted_ratings_df.clip(lower=0, upper=5)

# CSV 파일로 저장
predicted_ratings_df.to_csv('predicted_ratings_test.csv')
print("Predicted ratings saved to 'predicted_ratings_test.csv'.")

