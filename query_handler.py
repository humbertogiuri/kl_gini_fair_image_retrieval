from feature_extraction import FeatureExtractor
import cv2
import torch
import pandas as pd
import numpy as np
from utils import kl_divergence, normalize_vector, cosine_similarity
from topsis import topsis
import pandas as pd
from giniCoefficients import giniCoefficient, gini_coefficient_numpy
from sklearn.preprocessing import normalize

def hamming_distance(vector1, vector2):
    if len(vector1) != len(vector2):
        raise ValueError("Os vetores devem ter o mesmo comprimento")

    return np.sum(vector1 != vector2)

class QueryHandler:
    def __init__(self, db_path: str) -> None:
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        print(f'Using device: {self.device}')

        self.feture_extractor = FeatureExtractor()

        self.db_dataframe = pd.read_hdf(db_path)
        self.db_dataframe["ginis_prepros"], self.db_dataframe["ginis_raw"] = self._calculate_gini_index_many_to_many_similarity()

        print(self.db_dataframe.head(30))

    def _calculate_gini_index_many_to_many_similarity(self):
        features = self.db_dataframe.features.to_list()
        features_normalized = normalize(features, norm='l2')

        raw = np.array([gini_coefficient_numpy(row) for row in features_normalized])
        
        print(normalize([raw]))
        return giniCoefficient(features_normalized, normalize=True), normalize([raw])[0]

    def _get_clinical_data(self, name_image, path_csv_train="./CLINICAL/ndbufes_TaskIV_parsed_folders.csv", path_csv_test="./CLINICAL/ndbufes_TaskIV_parsed_test.csv"):
        tobacco_columns = ["tobacco_use_Yes", "tobacco_use_Former", "tobacco_use_No", "tobacco_use_Not informed"]
        alcohol_columns = ["alcohol_consumption_Yes", "alcohol_consumption_Former", "alcohol_consumption_No", "alcohol_consumption_Not informed"]
        
        all_clinical_columns = ["path"]

        all_clinical_columns.extend(tobacco_columns)
        all_clinical_columns.extend(alcohol_columns)

        #If "tobacco" in clinical_columns:
        #    all_clinical_columns.extend(tobacco_columns)
        
        #if "alcohol" in clinical_columns:
        #    all_clinical_columns.extend(alcohol_columns)
        
        
        #df_train = pd.read_csv(path_csv_train).drop(columns=['larger_size', 'TaskIV', 'folder'])                                                                                                                    
        #df_test = pd.read_csv(path_csv_test).drop(columns=['larger_size', 'TaskIV', '                                                                                                                                                                                                                               '])

        df_train = pd.read_csv(path_csv_train, usecols=all_clinical_columns)
        df_test = pd.read_csv(path_csv_test,  usecols=all_clinical_columns)

        query_series = df_test[df_test['path'] == name_image].squeeze()

        clinical_data_train = [(row['path'], row.drop('path').tolist()) for _, row in df_train.iterrows()]
        clinical_data_query = (query_series['path'], query_series.drop(labels=['path']).to_list())

        return clinical_data_query, clinical_data_train
    
    def apply_topsis(self, df):
        # Calcula os rankings usando TOPSIS
        a = df[['dist', 'kl_dist', 'cos_dist', 'ginis_raw', 'ginis_prepros']].to_numpy()
        #w = [0.33, 0.33, 0.33]
        w = [0.0, 0.0, 0.0, 0.0, 1.0]
        sign = [-1, -1, -1, -1, 1]

        return topsis(a, w, sign)

    def get_n_similares(self, query, n, file_name):
        query_features = self.feture_extractor.extract_features(img=query)
        #clinical_data_query, clinical_data_train = self._get_clinical_data(name_image=file_name)

        #df_data_train = pd.DataFrame(data=clinical_data_train, columns=["name", "clinical_data"])
        copy_df = self.db_dataframe.copy()

        #copy_df = pd.merge(copy_df, df_data_train, on='name', how='left')

        copy_df['dist'] = copy_df['features'].apply(
            lambda x: np.linalg.norm(query_features - x)
        )
        copy_df['cos_dist'] = copy_df['features'].apply(
            lambda x: cosine_similarity(query_features, x)
        )
        copy_df['kl_dist'] = copy_df['features'].apply(
            lambda x: kl_divergence(
                normalize_vector([query_features]), normalize_vector([x])
            )
        )
        
        #copy_df['ham_dist'] = copy_df.apply(
        #lambda row: hamming_distance(
        #    clinical_data_query[1],  # Informações clínicas da consulta
        #    row['clinical_data']     # Informações clínicas do banco de dados
        #), axis=1
        #   )

        best_topsis, all_topsis = self.apply_topsis(copy_df)

        copy_df['topsis'] = all_topsis

        return copy_df.sort_values(by=['topsis'], ascending=False).head(n)


if __name__ == '__main__':
    query_handler = QueryHandler('data_ndb.h5')
    
    query_path = (
        './PAD/PAD-BASE-SPLIT/test/Actinic Keratosis/PAT_368_2427_803.png'
    )
    query_image = cv2.imread(query_path)
    query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)

    similares = query_handler.get_n_similares(query_image, 5)
    print(similares)
