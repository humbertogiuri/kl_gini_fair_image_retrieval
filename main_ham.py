import os
import pandas as pd
from query_handler import QueryHandlerHAM

if __name__ == "__main__":
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

    handler = QueryHandlerHAM(
        db_path=path_pkl_train,
        metadata_path=path_metadata
    )

    results = {
        "name_query": [],
        "label_query": [],
        "sensitivy_group_query": [],
        "names_results": [],
        "labels_results": [],
        "sensitivity_groups_results": [],
    }


    df_sample = df_test.sample(n=100, random_state=42).reset_index()  
    for i, row in df_sample.iterrows():
        if i == 100:
            break

        name_query, features_query, label_query, sensitive_query = row["name"], row["features"], row["class"], row["sex"]

        names_results, sensitives_results, labels_results = handler.retrieve_similar_images_kl_fair_ranking(features_query)

        results["name_query"].append(name_query)
        results["label_query"].append(label_query)
        results["sensitivy_group_query"].append(sensitive_query)
        results["names_results"].append(names_results)
        results["labels_results"].append(labels_results)
        results["sensitivity_groups_results"].append(sensitives_results)
    
    df_results = pd.DataFrame(results)
    df_results.to_csv("results_ham.csv", index=False)

    accuracy = (df_results["label_query"] == df_results["labels_results"].apply(lambda x: x[0] if len(x) > 0 else None)).mean()
    print(f"Accuracy (top-1): {accuracy:.4f}")
