def run_pipeline(input_path, output_path):

    import pandas as pd
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.impute import SimpleImputer

    df = pd.read_csv(input_path)

    # target cleaning
    df = df.dropna(subset=['RainTomorrow'])
    df['RainTomorrow'] = df['RainTomorrow'].astype(str).str.strip()
    df = df[df['RainTomorrow'].isin(['Yes','No'])]
    df['RainTomorrow'] = df['RainTomorrow'].map({'No':0,'Yes':1})

    X = df.drop('RainTomorrow', axis=1)

    num_cols = X.select_dtypes(include=['float64']).columns
    cat_cols = X.select_dtypes(include=['object']).columns

    X[num_cols] = SimpleImputer(strategy='median').fit_transform(X[num_cols])
    X[cat_cols] = SimpleImputer(strategy='most_frequent').fit_transform(X[cat_cols])

    for col in cat_cols:
        X[col] = LabelEncoder().fit_transform(X[col])

    X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)

    df_final = X.copy()
    df_final['RainTomorrow'] = df['RainTomorrow'].values

    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, index=False)


if __name__ == "__main__":
    run_pipeline(
        "dataset_raw/weatherAUS.csv",
        "preprocessing/dataset_preprocessing/processed_data.csv"
    )