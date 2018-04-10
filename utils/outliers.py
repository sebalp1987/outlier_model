import utils.smirnov_grubbs as smirnov_grubbs
import pandas as pd
import numpy as np

class Outliers:

    def outliers_df(file_df, col_name, not_count_zero = True, just_count_zero = False, smirnov = True):


        file_df_col= file_df[col_name].dropna()
        file_df_col = file_df_col.convert_objects(convert_numeric=True)

        if not_count_zero == True:
            file_df_col= file_df_col[file_df_col > 0]
        if just_count_zero == True:
            file_df_col = file_df_col[file_df_col >= 0]

        file_df[col_name] = file_df[col_name].convert_objects(convert_numeric=True)

        # Marcamos las colas de la distribución 0.05 y 0.95
        list_outlier = []
        outlier_percentile = Outliers.percentile_based_outlier(file_df_col)
        for ax, func in zip(file_df_col, outlier_percentile):
            if func == True:  # True is outlier
                list_outlier.append(ax)
        list_outlier = set(list_outlier)

        name = str(col_name) + '_outlier_5_95'
        file_df[name] = pd.Series(0, index=file_df.index)
        file_df[name] = file_df.apply(
            lambda x: 1
            if x[col_name] in list_outlier
            else 0, axis=1)

        if smirnov == True:
            # Marcamos outliers con smirnov
            max_smirnov = smirnov_grubbs.max_test_outliers(file_df_col, alpha=0.10)
            min_smirnov = smirnov_grubbs.min_test_outliers(file_df_col, alpha=0.10)

            if max_smirnov:
                max_thresold = min(max_smirnov)
                name_max = str(col_name) + '_max_smirnov'
                file_df[name_max] = pd.Series(0, index=file_df.index)
                file_df[name_max] = file_df.apply(
                    lambda x: 1
                    if x[col_name] < max_thresold
                    else 0, axis=1)
            if min_smirnov:
                min_thresold = max(min_smirnov)
                name_min = str(col_name) + '_min_smirnov'
                file_df[name_min] = pd.Series(0, index=file_df.index)
                file_df[name_min] = file_df.apply(
                    lambda x: 1
                    if x[col_name] < min_thresold
                    else 0, axis=1)

        #MAD
        outliers_mad = Outliers.mad_based_outlier(file_df_col)
        list_outlier = []
        for ax, func in zip(file_df_col, outliers_mad):
            if func == True:  # True is outlier
                list_outlier.append(ax)
        list_outlier = set(list_outlier)
        name = str(col_name) + '_mad_outlier'
        file_df[name] = pd.Series(0, index=file_df.index)
        file_df[name] = file_df.apply(
            lambda x: 1
            if x[col_name] in list_outlier
            else 0, axis=1)


        return file_df



    def outliers_mad(file_df, col_name, not_count_zero = True, just_count_zero = False):
        from sklearn.preprocessing import scale

        file_df_col = file_df[col_name].dropna()
        file_df_col = pd.to_numeric(file_df_col, errors='coerce')

        file_df_col = file_df_col.fillna(file_df_col.median())

        if not_count_zero == True:
            file_df_col = file_df_col[file_df_col > 0]
        if just_count_zero == True:
            file_df_col = file_df_col[file_df_col >= 0]

        file_df[col_name] = pd.to_numeric(file_df[col_name], errors='coerce')
        file_df[col_name] = file_df[col_name].fillna(file_df[col_name].median())

        # file_df_col = scale(file_df_col)

        # MAD
        outliers_mad = Outliers.mad_based_outlier(file_df_col)
        list_outlier = []
        for ax, func in zip(file_df_col, outliers_mad):
            if func:  # True is outlier
                list_outlier.append(ax)
        list_outlier = set(list_outlier)
        name = str(col_name) + '_mad_outlier'
        file_df[name] = pd.Series(0, index=file_df.index)
        file_df[name] = file_df.apply(
            lambda x: 1
            if x[col_name] in list_outlier
            else 0, axis=1)

        return file_df


    def mad_based_outlier(points, thresh=3.5):
        if len(points.shape) == 1:
            points = points[:, None]

        median = np.median(points, axis=0)

        diff = np.sum((points - median) ** 2, axis=-1)

        diff = np.sqrt(diff)

        med_abs_deviation = np.median(diff)

        modified_z_score = 0.6745 * diff / med_abs_deviation

        return modified_z_score > thresh


    def percentile_based_outlier(data, threshold=95):
        diff = (100 - threshold) / 2.0
        minval, maxval = np.percentile(data, [diff, 100 - diff])
        return (data < minval) | (data > maxval)


    def outliers_test_values(file_df, base_df, col_name, not_count_zero=True, just_count_zero=False):
        from sklearn.preprocessing import scale
        # base_df
        base_df_col = base_df[col_name].dropna()
        base_df_col = pd.to_numeric(base_df_col, errors='coerce')

        base_df_col = base_df_col.fillna(base_df_col.median())

        if not_count_zero:
            base_df_col = base_df_col[base_df_col > 0]
        if just_count_zero:
            base_df_col = base_df_col[base_df_col >= 0]

        base_df[col_name] = pd.to_numeric(base_df[col_name], errors='coerce')
        base_df[col_name] = base_df[col_name].fillna(base_df[col_name].median())

        # test df
        print(file_df.columns.values.tolist())
        file_df_col = file_df[col_name].dropna()
        file_df_col = pd.to_numeric(file_df_col, errors='coerce')
        file_df_col = file_df_col.dropna()

        if not_count_zero == True:
            file_df_col = file_df_col[file_df_col > 0]
        if just_count_zero == True:
            file_df_col = file_df_col[file_df_col >= 0]


        # MAD
        median, med_abs_deviation = Outliers.mad_based_outlier_parameters(base_df_col)
        if len(file_df_col.shape) == 1:
            points = file_df_col[:, None]

        diff = np.sum((points - median) ** 2, axis=-1)
        diff = np.sqrt(diff)
        modified_z_score = 0.6745 * diff / med_abs_deviation

        outliers_mad = modified_z_score > 3.5

        list_outlier = []
        for ax, func in zip(file_df_col, outliers_mad):
            if func:  # True is outlier
                list_outlier.append(ax)
        list_outlier = set(list_outlier)
        name = str(col_name) + '_mad_outlier'
        file_df[name] = pd.Series(0, index=file_df.index)
        file_df[name] = file_df.apply(
            lambda x: 1
            if x[col_name] in list_outlier
            else 0, axis=1)

        return file_df

    def mad_based_outlier_parameters(points):
        if len(points.shape) == 1:
            points = points[:, None]

        median = np.median(points, axis=0)

        diff = np.sum((points - median) ** 2, axis=-1)

        diff = np.sqrt(diff)

        med_abs_deviation = np.median(diff)

        return median, med_abs_deviation