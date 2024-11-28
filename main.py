from src.load_and_prepare_data import load_data_to_pd

def main(path):
    load_data_to_pd(path)

if __name__ == "__main__":
    data_location = ".."
    file_name = "BPI_Challenge_2017.xes.gz"
    path = f"{data_location}\{file_name}"
    main(path)