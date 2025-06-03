from feature_extraction import FeatureExtractor
import cv2
import torch
import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
#from feature_similarity import (
    
#)

class QueryHandlerHAM:
    def __init__(self, db_path: str, metadata_path: str) -> None:
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.device = 'cpu'
        #print(f'Using device: {self.device}')

        self.feture_extractor = FeatureExtractor()
        self.db_dataframe = pd.read_pickle(db_path)
        self.metadata_df = pd.read_csv(metadata_path)

        self.db_dataframe = self.db_dataframe.merge(
            self.metadata_df[["image_id", "sex"]], 
            left_on="name", 
            right_on="image_id", 
            how="left"
        )

    
    def retrieve_similar_images_minimum_group_quota(self, query_features, top_k=5):
        """
        Retrieve the top_k most similar images to a given query, ensuring a minimum quota from each sensitive group.
        This function computes the cosine similarity between the query features and all images in the database.
        It then selects the top_k most similar images, enforcing that at least half of the selected images belong to the less-represented sensitive group (group_b), as determined by the 'sex' attribute in the database.
        If there are not enough images from group_b, the remaining slots are filled with images from group_a or the next most similar images.
        Args:
            query_features (np.ndarray): Feature vector representing the query image.
            top_k (int, optional): Number of similar images to retrieve. Defaults to 5.
        Returns:
            None: The function currently prints the similarities, selected indices, and their sensitive group labels.
        Raises:
            ValueError: If there are fewer than two unique sensitive groups in the database.
        """
        db_features = np.vstack(self.db_dataframe["features"].values)
        query_features = query_features.reshape(1, -1)

        db_sensitive_features = self.db_dataframe["sex"].to_numpy()

        groups_unique = list(set(db_sensitive_features))

        if len(groups_unique) < 2:
            raise ValueError("There must be at least two unique groups in the database for this method to work.")
        
        group_a, group_b = groups_unique[0], groups_unique[1]

        similarites = cosine_similarity(query_features, db_features).flatten()
        ranked_indices = np.argsort(-similarites)

        min_group_b_size = top_k // 2
        selected = list()
        count_group_b = 0
        
        for idx in ranked_indices:
            if len(selected) < top_k:
                if db_sensitive_features[idx] == group_b and count_group_b < min_group_b_size:
                    selected.append(idx)
                    count_group_b += 1

                elif db_sensitive_features[idx] == group_a or count_group_b >= min_group_b_size:
                    selected.append(idx)
        
        for idx in ranked_indices:
            if len(selected) >= top_k:
                break

            if idx not in selected:
                selected.append(idx)

        print(similarites[selected], selected, db_sensitive_features[selected])

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
    path_metadata = os.path.join("data", "HAM", "HAM10000_metadata.csv")

    df_test = pd.read_pickle(path_pkl_test)
    df_metadata = pd.read_csv(path_metadata)

    df_test = df_test.merge(
        df_metadata[["image_id", "sex"]], 
        left_on="name", 
        right_on="image_id", 
        how="left"
    )

    random_row = df_test.sample(n=1, random_state=42)
    random_feature = random_row["features"].values[0]

    query_handler = QueryHandlerHAM(path_pkl_train, path_metadata)
    

    result = query_handler.retrieve_similar_images_minimum_group_quota(random_feature)

