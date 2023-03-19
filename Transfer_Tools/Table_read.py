import numpy as np
import pandas as pd


def xlsread(file_address, header=0):
    value = pd.read_excel(file_address, header=header).values
    header = np.array(pd.read_excel(file_address, header=header).columns.tolist())
    full_table = np.insert(value, 0, header, axis=0)
    return full_table


def xlsread_value(file_address, usecols=None):
    return pd.read_excel(file_address, usecols=usecols).values


def get_rows(table, row_start, row_end=None):
    if row_end is None:
        row_end = row_start + 1
    return table[row_start: row_end, 0:]


def get_cols(table, col_start, col_end=None):
    if col_end is None:
        col_end = col_start + 1
    return table[0:, col_start: col_end]

