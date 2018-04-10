import pandas as pd
import numpy as np

class ReadCsv:

    def load_csv(file: str):
        """
        It load a file using read mode and a latin encoding.
        :return: A readed file
        """
        input_file = open(file, 'r', newline='', encoding='latin1')
        return input_file

    def processing_txt_without_header(file, separator=';', nan_val='?', header=False, change_decimal=False):
        """
        It loads a .txt file as a Dataframe and return a processed df without header
        """

        df = pd.read_csv(file, sep=separator, header=None, encoding='latin1', quotechar='"')
        if header == True:
            df = df[1:]

        if change_decimal == True:
            df = pd.read_csv(file, sep=separator, header=None, encoding='latin1', quotechar='"', decimal=',')

        df = df.replace(nan_val, np.nan)

        return df

    def load_blacklist(file):
        input_file = open(file, 'r', newline='\n', encoding = 'latin1')
        return input_file

