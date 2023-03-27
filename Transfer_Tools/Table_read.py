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


# 计算一元二次方程的解
def quadratic_equation(a, b, c):
    if a == 0:
        return -c / b
    else:
        delta = b ** 2 - 4 * a * c
        if delta >= 0:
            return (-b + np.sqrt(delta)) / (2 * a), (-b - np.sqrt(delta)) / (2 * a)
        else:
            return None, None


# 解一次常微分方程
def solve_ode(f, x0, y0, h, n):
    x = np.zeros(n)
    y = np.zeros(n)
    x[0] = x0
    y[0] = y0
    for i in range(n - 1):
        y[i + 1] = y[i] + h * f(x[i], y[i])
        x[i + 1] = x[i] + h
    return x, y
