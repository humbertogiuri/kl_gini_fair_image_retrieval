from feature_extraction import FeatureExtractor
import cv2
import torch
import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
from scipy.special import rel_entr
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

        return self.db_dataframe["name"].to_numpy()[selected], db_sensitive_features[selected]
    

    def retrieve_similar_images_wheighted_fair_ranking(self, query_features, top_k=5, alpha=0.8):
        """
        Retrieve the top-k most similar images to a query, applying a weighted fairness ranking based on a sensitive attribute.
        This function computes the cosine similarity between the query features and the database features, then combines this similarity
        with a fairness bonus that promotes images belonging to a specified sensitive group (e.g., 'sex'). The final ranking is determined
        by a weighted sum of similarity and fairness, controlled by the alpha parameter.

        The combined weighted score for each image is calculated as:
            combined_weighted_score = alpha * similarity + (1 - alpha) * fairness_bonus

        Args:
            query_features (np.ndarray): Feature vector representing the query image.
            top_k (int, optional): Number of top results to return. Defaults to 5.
            alpha (float, optional): Weighting factor between similarity and fairness (0 <= alpha <= 1). Defaults to 0.8.
        Returns:
            Tuple[np.ndarray, np.ndarray]: 
            - Array of names of the top-k retrieved images.
            - Array of sensitive attribute values corresponding to the top-k images.
        Raises:
            ValueError: If there are fewer than two unique groups in the sensitive attribute column.
        Note:
            This method assumes the sensitive attribute column is named 'sex' and exists in the database dataframe.
        """
        
        db_features = np.vstack(self.db_dataframe["features"].values)
        query_features = query_features.reshape(1, -1)

        db_sensitive_features = self.db_dataframe["sex"].to_numpy()
        groups_unique = list(set(db_sensitive_features))

        if len(groups_unique) < 2:
            raise ValueError("There must be at least two unique groups in the database for this method to work.")
        
        group_a, group_b = groups_unique[0], groups_unique[1]

        similarities = cosine_similarity(query_features, db_features).flatten()
        fairness_bonus = np.array([1 if g == group_b else 0 for g in db_sensitive_features])
        fairness_bonus = fairness_bonus / fairness_bonus.max()

        combined_wheighted_score = alpha * similarities + (1 - alpha) * fairness_bonus

        ranked_indices = np.argsort(-combined_wheighted_score)

        return self.db_dataframe["name"].to_numpy()[ranked_indices][:top_k], db_sensitive_features[ranked_indices][:top_k]

    def retrieve_similar_images_kl_fair_ranking(self, query_features, top_k=5, alpha=0.5):
        """
        Esta função recupera as imagens mais similares a uma consulta, balanceando similaridade e equidade entre dois grupos sensíveis usando uma métrica baseada em divergência KL.

        A função busca garantir que o conjunto de imagens retornadas seja não apenas similar à consulta (com base na similaridade do cosseno), mas também justo em relação à distribuição dos grupos sensíveis (por exemplo, sexo). Para isso, ela utiliza a divergência KL para medir o quão distante a distribuição dos grupos no conjunto selecionado está da distribuição ideal (balanceada).

        Funcionamento:
            1. Calcula a similaridade do cosseno entre a imagem de consulta e todas as imagens do banco.
            2. Seleciona os 30 candidatos mais similares.
            3. Gera todas as combinações possíveis de tamanho `top_k` entre esses candidatos.
            4. Para cada combinação, calcula:
            - A média das similaridades.
            - A divergência KL entre a distribuição real dos grupos sensíveis e a distribuição ideal (balanceada).
            - Um escore combinado: média da similaridade menos `alpha` vezes a divergência KL.
            5. Retorna a combinação com o maior escore combinado.

        Parâmetros:
            query_features (np.ndarray): Vetor de características da imagem de consulta.
            top_k (int, opcional): Número de imagens a serem recuperadas. Padrão é 5.
            alpha (float, opcional): Parâmetro de balanceamento entre similaridade e equidade. Valores maiores priorizam mais a equidade. Padrão é 0.5.

        Retorno:
            Tuple[np.ndarray, np.ndarray]:
            - Array com os nomes das imagens selecionadas.
            - Array com os rótulos dos grupos sensíveis das imagens selecionadas.

        Exceções:
            ValueError: Caso haja menos de dois grupos sensíveis únicos no banco de dados.

        Observações:
            - O atributo sensível considerado é 'sex'.
            - O método pode ser computacionalmente custoso para valores altos de `top_k` devido à geração de combinações.
        """
        db_features = np.vstack(self.db_dataframe["features"].values)
        query_features = query_features.reshape(1, -1)

        db_sensitive_features = self.db_dataframe["sex"].to_numpy()
        groups_unique = list(set(db_sensitive_features))

        if len(groups_unique) < 2:
            raise ValueError("There must be at least two unique groups in the database for this method to work.")
        
        group_a, group_b = groups_unique[0], groups_unique[1]

        similarities = cosine_similarity(query_features, db_features).flatten()
        candidate_indices = np.argsort(-similarities)[:30]

        ideal_dist = np.array([0.5, 0.5])  # 50% A, 50% B
        best_set = None
        best_score = -np.inf

        for subset in combinations(candidate_indices, top_k):
            subset = list(subset)
            subset_sensitive = db_sensitive_features[subset]

            group_a_count = (subset_sensitive == group_a).sum()
            group_b_count = (subset_sensitive == group_b).sum()

            real_dist = np.array([group_a_count, group_b_count]) / top_k

            epsilon = 1e-10
            real_dist = np.clip(real_dist, epsilon, 1)
            ideal_dist = np.clip(ideal_dist, epsilon, 1)

            kl_div = np.sum(rel_entr(real_dist, ideal_dist))

            avg_similarity = np.mean(similarities[subset])
            score = avg_similarity - alpha * kl_div

            if score > best_score:
                best_score = score
                best_set = subset

        return self.db_dataframe["name"].to_numpy()[best_set], db_sensitive_features[best_set]
    

    def retrieve_similar_images_gini_fair_ranking(self, query_features, top_k=5, alpha=0.5):
        db_features = np.vstack(self.db_dataframe["features"].values)
        query_features = query_features.reshape(1, -1)

        db_sensitive_features = self.db_dataframe["sex"].to_numpy()
        groups_unique = list(set(db_sensitive_features))

        if len(groups_unique) < 2:
            raise ValueError("There must be at least two unique groups in the database for this method to work.")
        
        group_a, group_b = groups_unique[0], groups_unique[1]

        similarities = cosine_similarity(query_features, db_features).flatten()
        candidate_indices = np.argsort(-similarities)[:30]

        best_set = None
        best_score = -np.inf

        for subset in combinations(candidate_indices, top_k):
            subset = list(subset)
            subset_sensitive = db_sensitive_features[subset]

            group_a_count = (subset_sensitive == group_a).sum()
            group_b_count = (subset_sensitive == group_b).sum()

            group_counts = np.array([group_a_count, group_b_count])
            gini_index = self.gini_index(group_counts)
            
            avg_similarity = np.mean(similarities[subset])
            score = avg_similarity - alpha * gini_index

            if score > best_score:
                best_score = score
                best_set = subset
            
        return self.db_dataframe["name"].to_numpy()[best_set], db_sensitive_features[best_set]



    def gini_index(self, group_counts):
        values = np.array(group_counts)
        if np.sum(values) == 0:
            return 0
        sorted_values = np.sort(values)
        n = len(values)
        cumulative = np.cumsum(sorted_values)
        gini = (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n
        return gini


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
    

    names, groups = query_handler.retrieve_similar_images_minimum_group_quota(random_feature)
    print("Minimum Group Quota Results:")
    for name, group in zip(names, groups):
        print(f"Image: {name}, Group: {group}")
    print("-" * 40)

    names, groups = query_handler.retrieve_similar_images_wheighted_fair_ranking(random_feature)
    print("Weighted Fair Ranking Results:")
    for name, group in zip(names, groups):
        print(f"Image: {name}, Group: {group}")
    print("-" * 40)

    names, groups = query_handler.retrieve_similar_images_kl_fair_ranking(random_feature)
    print("KL Fair Ranking Results:")
    for name, group in zip(names, groups):
        print(f"Image: {name}, Group: {group}")
    print("-" * 40)

    names, groups = query_handler.retrieve_similar_images_gini_fair_ranking(random_feature)
    print("Gini Fair Ranking Results:")
    for name, group in zip(names, groups):
        print(f"Image: {name}, Group: {group}")
    print("-" * 40)

