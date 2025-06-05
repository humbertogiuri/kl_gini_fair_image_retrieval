import os
import pandas as pd
from query_handler import QueryHandlerHAM

POSSIBLE_RANKING_METHODS = [
    "minimum_group_quota",
    "wheighted_fair",
    "kl_fair",
    "gini_fair",
    "entropy_fair"
]


def run_test_ranking_method(handler, method, df):
    results = {
        "name_query": [],
        "label_query": [],
        "sensitivy_group_query": [],
        "names_results": [],
        "labels_results": [],
        "sensitivity_groups_results": [],
    }

    for i, row in df.iterrows():
        print(f"Processing row {i+1}/{len(df)}: {row['name']}")

        name_query, features_query, label_query, sensitive_query = row["name"], row["features"], row["class"], row["sex"]

        if method == "kl_fair":
            names_results, sensitives_results, labels_results = handler.retrieve_similar_images_kl_fair_ranking(features_query)

        elif method == "minimum_group_quota":
            names_results, sensitives_results, labels_results = handler.retrieve_similar_images_minimum_group_quota(features_query)
        
        elif method == "wheighted_fair":
            names_results, sensitives_results, labels_results = handler.retrieve_similar_images_wheighted_fair_ranking(features_query)
        
        elif method == "gini_fair":
            names_results, sensitives_results, labels_results = handler.retrieve_similar_images_gini_fair_ranking(features_query)
        
        elif method == "entropy_fair":
            names_results, sensitives_results, labels_results = handler.retrieve_similar_images_entropy_fair_ranking(features_query)
        
        else:
            raise ValueError(f"Unknown ranking method: {method}")
        
        results["name_query"].append(name_query)
        results["label_query"].append(label_query)
        results["sensitivy_group_query"].append(sensitive_query)
        results["names_results"].append(names_results)
        results["labels_results"].append(labels_results)
        results["sensitivity_groups_results"].append(sensitives_results)
    
    return results

if __name__ == "__main__":
    path_pkl_train = os.path.join("data", "HAM", "ham_train_features.pkl")
    path_pkl_test = os.path.join("data", "HAM", "ham_test_features.pkl")
    path_metadata = os.path.join("data", "HAM", "HAM10000_metadata.csv")
    ranking_method = "gini_fair"

    handler = QueryHandlerHAM(
        db_path=path_pkl_train,
        metadata_path=path_metadata
    )

    df_test = pd.read_pickle(path_pkl_test)
    df_metadata = pd.read_csv(path_metadata)

    df_test = df_test.merge(
        df_metadata[["image_id", "sex"]], 
        left_on="name", 
        right_on="image_id", 
        how="left"
    )
    
    df_sample = df_test.sample(n=100, random_state=42).reset_index()  

    for method in POSSIBLE_RANKING_METHODS:
        print(f"Running test for ranking method: {method}")
        path_output = os.path.join("data", "HAM", "results", f"ham_{method}.csv")

        results = run_test_ranking_method(handler=handler, method=method, df=df_sample)

        df_results = pd.DataFrame(results)
        df_results.to_csv(path_output, index=False)
