from feature_extraction import FeatureExtractor
import cv2
import torch
import pandas as pd
import numpy as np
import os

class QueryHandlerHAM:
    def __init__(self, db_path: str) -> None:
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.device = 'cpu'
        #print(f'Using device: {self.device}')

        self.feture_extractor = FeatureExtractor()
        self.db_dataframe = pd.read_pickle(db_path)

        print(self.db_dataframe.head(30))

    def calculate_all_distances(self, query):
        query_features = self.feture_extractor.extract_features(img=query)
        copy_df = self.db_dataframe.copy()

        # Calcular todas as métricas
        copy_df['l2'] = copy_df['features'].apply(
            lambda x: np.linalg.norm(query_features - x))
        
        copy_df['kl'] = copy_df['features'].apply(
            lambda x: kl_divergence(normalize_vector(query_features), normalize_vector(x)))
        
        copy_df['manhattan'] = copy_df['features'].apply(
            lambda x: minkowski_distance(query_features, x, p=1))
        
        copy_df['lorentzian'] = copy_df['features'].apply(
            lambda x: lorentzian_distance(query_features, x))
        
        copy_df['canberra'] = copy_df['features'].apply(
            lambda x: canberra_distance(query_features, x))
        
        copy_df['cosine'] = copy_df['features'].apply(
            lambda x: cosine_distance(normalize_vector(query_features), normalize_vector(x)))
        
        copy_df['hellinger'] = copy_df['features'].apply(
            lambda x: hellinger_distance(query_features, x))
        
        copy_df['squared_chi'] = copy_df['features'].apply(
            lambda x: squared_chi_squared(query_features, x))
        
        copy_df['jensen_shannon'] = copy_df['features'].apply(
            lambda x: jensen_shannon_divergence(query_features, x))
        
        copy_df['vicis_symmetric'] = copy_df['features'].apply(
            lambda x: vicis_symmetric(query_features, x))
        
        copy_df['hassanat'] = copy_df['features'].apply(
            lambda x: hassanat_distance(query_features, x))

        return copy_df


if __name__ == '__main__':
    path_pkl_train = os.path.join("data", "HAM", "ham_train_features.pkl")
    path_pkl_test = os.path.join("data", "HAM", "ham_test_features.pkl")

    query_handler = QueryHandlerHAM(path_pkl_train)
    exit()
    query_path = (
        './data/NDB-ORIGINAL/test/noDysplasia/0022.png'
    )
    query_image = cv2.imread(query_path)
    query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)

    similares = query_handler.calculate_all_distances(query_image)
    print(similares[['class', 'lorentzian']])
