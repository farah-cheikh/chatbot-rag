import pandas as pd
def check_quality(documents):

    df = pd.DataFrame(documents)
    df["characters"] = df["text"].str.len()
    df["words"] = df["text"].str.split().str.len()

    print(" rapporrt")
    print("nombre de documents :", len(df))
    print("documents vides :", (df["text"] == "").sum())
    print("doublons :", df["text"].duplicated().sum())

    print("\nstatistiques :")
    print(df[["characters", "words"]].describe())

    return df