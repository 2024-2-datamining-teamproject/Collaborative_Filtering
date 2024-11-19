import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

# predicted_ratings.csv와 ratings.csv 로드
predicted_ratings_df = pd.read_csv('predicted_ratings.csv', index_col=0)
ratings = pd.read_csv('ratings.csv')  # ratings.csv에는 'userId', 'movieId', 'rating' 컬럼이 있어야 합니다.

# predicted_ratings 데이터 준비
predicted_ratings = predicted_ratings_df.values
movie_ids = predicted_ratings_df.columns.astype(int)  # movieId를 정수로 변환
user_ids = predicted_ratings_df.index.astype(int)  # userId를 정수로 변환

# 사용자-아이템 평점 행렬 생성
ratings_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')
ratings_matrix = ratings_matrix.reindex(columns=movie_ids, fill_value=np.nan)  # predicted_ratings와 movieId 동기화
original_ratings_matrix = ratings_matrix.copy()  # NaN을 유지한 원본 행렬

# NaN을 0으로 채움 (KNN 모델 학습용)
ratings_matrix = ratings_matrix.fillna(0)


def recommend_by_prediction(user_id, top_n=30):
    """
    predicted_ratings.csv 기반으로 특정 사용자가 평가하지 않은 영화 중 예상 평점이 높은 상위 N개 추천
    """
    # user_id에 해당하는 인덱스 찾기
    user_idx = np.where(user_ids == user_id)[0][0]
    user_ratings = predicted_ratings[user_idx]
    
    # 사용자가 평가한 영화 제외
    rated_movies = original_ratings_matrix.loc[user_id][original_ratings_matrix.loc[user_id].notna()].index
    unrated_movies = [movie for movie in movie_ids if movie not in rated_movies]
    
    # 예상 평점 정렬
    recommendations = {movie: user_ratings[np.where(movie_ids == movie)[0][0]] for movie in unrated_movies}
    sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
    
    return [int(movie_id) for movie_id, _ in sorted_recommendations[:top_n]]


# def recommend_by_knn(user_id, top_n=30):
#     """
#     KNN을 통해 유사한 사용자가 높게 평가한 것으로 예측된 영화를 추천
#     """
#     knn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=5, n_jobs=-1)
#     knn_model.fit(predicted_ratings)
#     user_idx = np.where(user_ids == user_id)[0][0]
#     _, indices = knn_model.kneighbors([predicted_ratings[user_idx]], n_neighbors=5)
#     similar_users = user_ids[indices.flatten()]
#     recommendations = {}
#     for similar_user in similar_users:
#         similar_user_idx = np.where(user_ids == similar_user)[0][0]
#         similar_user_ratings = predicted_ratings[similar_user_idx]
#         for movie_idx, rating in enumerate(similar_user_ratings):
#             movie_id = movie_ids[movie_idx]
#             if pd.isna(original_ratings_matrix.loc[user_id, movie_id]):
#                 if movie_id not in recommendations:
#                     recommendations[movie_id] = rating
#                 else:
#                     recommendations[movie_id] += rating
#     recommended_movies = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
#     return [int(movie_id) for movie_id, _ in recommended_movies[:top_n]]

def recommend_by_knn(user_id, top_n=30):
    """
    KNN을 통해 실제 평점(ratings.csv)을 기반으로 유사한 사용자가 높게 평가한 영화를 추천
    """
    # KNN 모델 학습
    knn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=5, n_jobs=-1)
    knn_model.fit(ratings_matrix.values)  # 실제 평점 데이터 사용
    
    # user_id에 해당하는 인덱스 찾기
    user_idx = ratings_matrix.index.get_loc(user_id)
    
    # 유사한 사용자 찾기
    _, indices = knn_model.kneighbors([ratings_matrix.iloc[user_idx].values], n_neighbors=5)
    similar_users = ratings_matrix.index[indices.flatten()]
    
    # 유사한 사용자의 영화 추천
    recommendations = {}
    for similar_user in similar_users:
        similar_user_ratings = ratings_matrix.loc[similar_user]
        
        # 현재 사용자가 평가하지 않은 영화만 추천 후보로 추가
        for movie_id, rating in similar_user_ratings.items():
            if pd.isna(original_ratings_matrix.loc[user_id, movie_id]) and not pd.isna(rating):
                if movie_id not in recommendations:
                    recommendations[movie_id] = rating
                else:
                    recommendations[movie_id] += rating  # 누적 평점
    
    # 예상 평점 정렬
    recommended_movies = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
    
    # 상위 N개 영화 반환 (movieId만 반환)
    return [int(movie_id) for movie_id, _ in recommended_movies[:top_n]]


# 예제 실행
user_id = 2  # 추천을 받을 사용자 ID
print("Recommendations for User 1 by Prediction:", recommend_by_prediction(user_id=user_id))
print("Recommendations for User 1 by KNN:", recommend_by_knn(user_id=user_id))
