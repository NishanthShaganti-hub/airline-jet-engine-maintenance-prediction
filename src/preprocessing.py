import pandas as pd

def load_and_preprocess_data():
    df=pd.read_csv(
        "data/train_FD001.txt",
        sep=r"\s+",
        header=None
    )
    
    columns=['engine_id', 'cycle']
    columns += [f'operating_settings_{i}' for i in range(1,4)]
    columns += [f'sensor_{i}' for i in range(1,22)]

    df.columns = columns

    max_cycles = df.groupby("engine_id")["cycle"].max()

    df = df.merge(
        max_cycles,
        on="engine_id",
        suffixes=("", "_max")
    )

    df["RUL"] = df["cycle_max"] - df["cycle"]
    return df



if __name__ == "__main__":

    df = load_and_preprocess_data()
    
    print("Preprocessing completed!")
    print("Shape:", df.shape)


    print(df.head())
    print(df.shape)