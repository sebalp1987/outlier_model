import pandas as pd
import STRING
from . import write_csv
import numpy as np
import csv

class DfUtils:

    def load_df_file(self):
        """
        :return: It returns a CSV as a DF
        """
        file_load = pd.read_csv(self, delimiter=';', encoding='latin1')
        return file_load

    def statistic_df(self, output = False):

        """
        This returns a describe() and a null analysis of the file used as input.
        Also, if output = True, it returns two CSV files in doc_output\statistics
        wit names describe.csv and null.csv
        """


        print('DESCRIBE-------------------------')
        describe_df = self.describe(include = 'all')
        print(describe_df)
        print(' ')

        print('NULL VALUES-----------------------')

        null = self.isnull().sum()
        print(null)

        print(' ')
        print('INFO-----------------------------')
        print(self.info())
        print(' ')

        if output == True:
            import os
            import time

            os.chdir(STRING.path_project)


            timestr = time.strftime('%Y%m%d-%H%M%S')

            name_file = 'doc_output\statistics\\' + timestr +'_describe.csv'
            null_file = 'doc_output\statistics\\' + timestr + '_null.csv'
            print('File Exported as ' + name_file)

            describe_df.to_csv(name_file, sep =';', header= True, index=True,encoding='latin1')
            null.to_csv(null_file, sep=';', header=True, index=True, encoding='latin1')

    def del_var(names: list, df):
        """
        It takes a list of names and drop each of them.

        :param df: input Dataframe
        :return: Dataframe without the columns listed.
        """

        df = df.drop(names, axis = 1)

        return df

    def df_fillna(names: list, df, value):
        """
        Fill each cell nan value of the list name with the values 'value'

        :param df: input dataframe
        :param value: Value used as inputation value
        :return: Dataframe with values replaced
        """

        for i in names:
            df[i] = df[i].fillna(value)

        return df

    def string_categoric(names: list, var_grouped: str, df, type_agg = 'count'):
        """
        It takes a list of variables and print a groupby by a choosen var_grouped variable.

        :param var_grouped: Variables that want to be showed in the groupby
        :param df: input Dataframe
        :param type_agg: Operation type. 'count' as default.
        :return: It prints the result

        Eg:
        DfUtils.string_categoric(names, 'ID_DOSSIER', file)
        This will take each name from the list 'names' and will be groupby 'ID_DOSSIER'
        with an operation default = 'count' using 'file' Dataframe as inpùt.

        """

        for i in names:
            df = df.drop_duplicates(subset = [var_grouped], keep ='last')
            count_motivo = df.groupby(i)[var_grouped].agg([type_agg])
            print(count_motivo)

    def values_variables(df, path):
        """
        It takes a DF and it returns the unique values that take each column.
        :return: Return a csv to the path indicated with each column
        """

        list_values = []

        for i in df.columns:
            if (df[i].dtype == np.float64 or df[i].dtype == np.int64):
                values = df[i].unique()

                list_values.append([i, values])

        write_csv.WriteCsv.write_csv(list_values, path)

    def processing_file(file, delimiter=',', nan_values='?'):
        """
        It loads and gives a csv format to the BOTTLE raw file
        """

        list_id = []

        with file as csvfile:
            reader_line = csv.reader(csvfile, delimiter= delimiter, quotechar='"', quoting=csv.QUOTE_ALL)
            for lines in reader_line:
                list_id.append(lines)

        df = pd.DataFrame.from_records(list_id, index=None)

        new_header = df.iloc[0]
        new_header = new_header.str.replace('\"', '')
        new_header = new_header.str.replace(' ', '')
        df = df[1:]
        df = df.rename(columns=new_header)
        df = df.replace(nan_values, np.nan)

        return df


    def processing_file_without_header(file, delimiter =',', first_row = True):
        """
        It loads and gives a csv format to the BOTTLE raw file
        """

        list_id = []

        with file as csvfile:
            reader_line = csv.reader(csvfile, delimiter= delimiter, quotechar='"', quoting=csv.QUOTE_ALL)
            for lines in reader_line:
                list_id.append(lines)

        df = pd.DataFrame.from_records(list_id, index=None)
        if first_row == False:
            df = df[1:]
        df = df.replace('?', np.nan)

        return df