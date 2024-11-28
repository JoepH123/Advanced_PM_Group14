from src.load_and_prepare_data import load_data_xes_to_pd
import pandas as pd
import os

def main(data_location, file_name):
    file_name_no_ext = file_name.split(".")[0]
    file_path_gzip = f"{data_location}\{file_name_no_ext}.gzip"
    if not os.path.isfile(file_path_gzip):
        file_path_xes = f"{data_location}\{file_name}"
        df = load_data_xes_to_pd(file_path_xes)
    else:
        df = pd.read_parquet(file_path_gzip)
    print(df)
    return df


if __name__ == "__main__":
    data_location = "data"
    file_name = "BPI_Challenge_2017.xes.gz"
    main(data_location, file_name)