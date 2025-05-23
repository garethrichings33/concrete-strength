import pandas as pd


def get_data_from_csv(filename):
    with open(filename, "r") as file:
        dataframe = pd.read_csv(filename)

    return dataframe


def extract_raw_data(filename, path="./"):

    file = path
    if file[-1] != "/":
        file += "/"
    file += filename

    dataframe = get_data_from_csv(file)

    return dataframe
