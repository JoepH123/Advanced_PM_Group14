"""Module docstring"""

import pm4py


def load_data_to_pd(path):
    log = pm4py.read_xes(path)
    pd = pm4py.convert_to_dataframe(log)
    return pd