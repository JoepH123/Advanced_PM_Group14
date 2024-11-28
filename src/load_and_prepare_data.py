"""Module docstring"""

import pm4py


def load_data_xes_to_pd(path):
    log = pm4py.read_xes(path)
    df = pm4py.convert_to_dataframe(log)
    df.to_parquet("data/BPI_Challenge_2017.gzip")  # Save to gzip so that we can more quickly open to pandas dataframe
    return df