import pandas as pd
from sklearn.feature_selection import VarianceThreshold
import numpy as np
import matplotlib.pyplot as plot

class process_utils:

    def variance_threshold(self: pd.DataFrame, cp, fecha, threshold=0.0):
        """        
        VarianceThreshold is a simple baseline approach to feature selection. It removes all features whose variance
        doesn’t meet some threshold. By default, it removes all zero-variance features, i.e.
        features that have the same value in all samples.
        As an example, suppose that we have a dataset with boolean features,
        and we want to remove all features that are either one or zero (on or off) in more than 80% of the samples.
        """
        column_names = self.columns.values.tolist()
        key_variables = ['id_siniestro', 'id_poliza', 'cod_filiacion'] + cp + fecha
        removed_var = []
        for i in key_variables:
            try:
                column_names.remove(i)
                removed_var.append(i)
            except:
                pass

        append_names = []
        for i in column_names:
            self_i = self[[i]]
            self_i = self_i.apply(pd.to_numeric, errors='coerce')
            self_i = self_i.dropna(how='any', axis=0)
            selection = VarianceThreshold(threshold=threshold)
            try:
                selection.fit(self_i)
                features = selection.get_support(indices=True)
                features = self_i.columns[features]
                features = [column for column in self_i[features]]
                selection = pd.DataFrame(selection.transform(self_i), index=self_i.index)
                selection.columns = features
                append_names.append(selection.columns.values.tolist())
            except:
                pass

        append_names = [item for sublist in append_names for item in sublist]
        append_names = list(set(append_names))
        self = self[removed_var + append_names]
        return self

    def append_df(df_base: pd.DataFrame, df_add: pd.DataFrame, on_var: str='id_siniestro', on_var_type=int, how='left'):
        """
        It appends a dataframe based on 'id_siniestro' using join left. Also it returns the column names of the new
        dataframe so in the next step we can evaluate missing values
        
        :param df_add: The new DataFrame
        :param on_var: The key column
        :param on_var_type: The type of the key column
        :return:  df_base + df_add, df_add column names
        """
        print('addding... ')
        print(df_add.columns.values.tolist())
        print('initial shape ', df_base.shape)
        df_base[on_var] = df_base[on_var].map(on_var_type)
        df_add[on_var] = df_add[on_var].map(on_var_type)
        base_columns = df_base.columns.values.tolist()
        base_columns.remove(on_var)
        cols_to_use = df_add.columns.difference(base_columns)

        df_base = pd.merge(df_base, df_add[cols_to_use], how=how, on=on_var)
        df_add_cols = df_add.columns.values.tolist()
        df_add_cols.remove(on_var)
        print('final shape ', df_base.shape)
        return df_base, df_add_cols

    def fillna_by_bottle(df_base: pd.DataFrame, df_add_cols: list, cp, fecha, fill_value=-1):
        """
        Using append_df, we get the column names added. Then we will fillna only if the whole columns are NaN
        after they have been appended.
        
        :param df_add_cols: the column names just added
        :param fill_value: The fillna value we choose
        :return: Dataframe with the fillna process
        """

        key_variables = cp + fecha
        removed_var = []
        for i in key_variables:
            if i in df_base:
                removed_var.append(i)

        df_removed = df_base[removed_var]

        for i in key_variables:
            if i in df_base:
                del df_base[i]
            if i in df_add_cols:
                df_add_cols.remove(i)

        condition_nan_values = len(df_add_cols)
        print('condition to fill', condition_nan_values)
        for i in df_add_cols:
            df_base[i] = pd.to_numeric(df_base[i], errors='coerce')

        # We create a variable that count how many NAN values are in the selected columns df_add_cols
        df_base['count_NAN'] = df_base[df_add_cols].isnull().sum(axis=1)

        print('condition is get', df_base['count_NAN'].mode())
        # We make a fillna only if 'count_NAN' is exactly the number of new columns

        df_base = df_base.apply(lambda x: x.fillna(fill_value) if x['count_NAN'] == condition_nan_values
                                                                  else x, axis=1)
        del df_base['count_NAN']

        df_base = pd.concat([df_base, df_removed], axis=1)
        return df_base

    def delete_row_df_by_name(df_base: pd.DataFrame, del_name: str = 'MIGRA'):
        """
        If the del_name is contained in a column name it will delete the row which is == 1 from the dataframe

        :param del_name: String we want to search for deleting porpose.
        :return: df_base without the rows found
        """

        col_names = df_base.columns.values.tolist()
        col_names = [f for f in col_names if del_name in f]

        for i in col_names:
            rows_0 = len(df_base.index)
            df_base = df_base[df_base[i] != 1]
            del df_base[i]
            print(i + ' was Deleted')

            rows_1 = len(df_base.index)
            print(rows_0 - rows_1, ' rows were deleted')

        return df_base

    def fillna_multioutput(df: pd.DataFrame, not_consider: list = ['id_siniestro'], n_estimator=300,
                           max_depth=500, n_features=3):
        """
        Multioutput regression used for estimating NaN values columns. 
        :return: df with multioutput fillna
        """

        from sklearn.ensemble import RandomForestRegressor
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.model_selection import train_test_split


        # First we determine which columns have NaN values are which not

        jcols = df.columns[df.isnull().any()].tolist()
        icols = df.columns.values.tolist()
        for i in jcols:
            icols.remove(i)

        # We evaluate here which rows are null value. This returns a boolean for each row
        notnans = df[jcols].notnull().all(axis=1)

        # We create a df with not nan values which will be used as train-test in a supervised model
        df_notnans = df[notnans]
        pd.set_option('display.max_rows', 350000)
        print(icols)
        print(jcols)

        # We create a train-test set with X = icols values that do not have null values. And we try to estimate
        # the values of jcols (the columns with NaN). Here we are not considering the NaN values so we can estimate
        # as a supervised model the nan_cols. And finally, we apply the model estimation to the real NaN values.
        X_train, X_test, y_train, y_test = train_test_split(df_notnans[icols], df_notnans[jcols],
                                                            train_size=0.70,
                                                            random_state=42)

        n_estimator = n_estimator
        max_features = (round((len(df_notnans.columns))/n_features))
        min_samples_leaf = round(len(df_notnans.index)*0.005)
        min_samples_split = min_samples_leaf * 10
        max_depth = max_depth

        print('RANDOM FOREST WITH: ne_estimator='+str(n_estimator) + ', max_features=' + str(max_features) +
              ', min_samples_leaf=' + str(min_samples_leaf) + ', min_samples_split='
              + str(min_samples_split) + ', max_depth=' + str(max_depth))

        regr_multirf = MultiOutputRegressor(RandomForestRegressor(n_estimators=n_estimator, max_depth=max_depth,
                                                                  random_state=42, verbose=1,
                                                                  max_features=max_features,
                                                                  min_samples_split=min_samples_split,
                                                                  min_samples_leaf=min_samples_leaf))

        # We fit the model deleting variables that must not be included to do not have endogeneity (for example FRAUD
        # variable)
        regr_multirf.fit(X_train.drop(not_consider, axis=1), y_train)

        # We get R2 to determine how well is explaining the model
        score = regr_multirf.score(X_test.drop(not_consider, axis=1), y_test)
        print('R2 model ', score)

        # Now we bring the complete column dataframe with NaN row values
        df_nans = df.loc[~notnans].copy()
        df_not_nans = df.loc[notnans].copy()
        # Finally what we have to do is to estimate the NaN columns from the previous dataframe. For that we use
        # multioutput regression. This will estimate each specific column using Random Forest model. Basically we
        # need to predict dataframe column NaN values for each row in function of dataframe column not NaN values.
        df_nans[jcols] = regr_multirf.predict(df_nans[icols].drop(not_consider, axis=1))

        df_without_nans = pd.concat([df_nans, df_not_nans], axis=0, ignore_index=True)

        df = pd.merge(df, df_without_nans, how='left', on='id_siniestro', suffixes=('', '_y'))

        for i in jcols:
            df[i] = df[i].fillna(df[i+'_y'])
            del df[i+'_y']

        filter_col = [col for col in df if col.endswith('_y')]
        for i in filter_col:
            del df[i]

        return df

    def robust_scale(df:pd.DataFrame, quantile_range=(25.0, 75.0)):
        """
        Scale features using statistics that are robust to outliers.
        This Scaler removes the median and scales the data according to the quantile range 
        (defaults to IQR: Interquartile Range). The IQR is the range between the 1st quartile (25th quantile) 
        and the 3rd quartile (75th quantile).
        :return: scaled df
        """
        from sklearn.preprocessing import RobustScaler

        robust_scaler = RobustScaler(quantile_range=quantile_range)

        df_cols = df.columns.values.tolist()
        df_cols.remove('id_siniestro')

        params = []

        for i in df_cols:
            X= df[[i]]
            df[i] = robust_scaler.fit_transform(X)
            center = robust_scaler.center_
            scale = robust_scaler.scale_
            center = float(center[0])
            scale = float(scale[0])
            params.append([str(i), center, scale])

        params = pd.DataFrame(params, columns=['column_name', 'center', 'scale'])

        return df, params

    def pca_reduction(df: pd.DataFrame, show_plot=False, variance=95.00):
        """
        This automatically calcualte a PCA to df taking into account the 95% of the dataset explained variance
        :param show_plot: Threshold Variance Plot
        :param variance: Dataset variance limit to consider in the PCA.
        :return: PCA df
        """
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import scale

        siniestro_df = df[['id_siniestro', 'TEST']]
        del df['id_siniestro']
        del df['TEST']
        columns = len(df.columns)
        X= scale(df)
        pca = PCA(whiten=True, svd_solver='randomized', n_components=columns)

        pca.fit(X)
        cumsum = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4) * 100)
        cumsum = list(cumsum)
        var = [value for value in cumsum if value <= variance]
        pca_components = len(var)


        if show_plot == True:
            plot.plot(cumsum)
            plot.title('Explained Variance Ratio (PCA)')
            plot.xlabel('N features')
            plot.axvline(x=pca_components, color='r', ls='--')
            plot.ylabel('Dataset Variance', rotation=90)
            import datetime
            import os
            DAY = datetime.datetime.today().strftime('%Y-%m-%d')
            path_probabilidad_day = 'final_files\\' + str(DAY) + '\\'
            os.makedirs(os.path.dirname(path_probabilidad_day), exist_ok=True)
            plot.savefig(path_probabilidad_day + 'pca.png')
            plot.close()

        print('PCA Components ', pca_components)

        pca = PCA(n_components=pca_components, whiten=True, svd_solver='randomized')
        df = scale(df)
        pca.fit(df)
        df = pca.fit_transform(df)
        df = pd.DataFrame(df)

        df = pd.concat([df, siniestro_df], axis=1)

        return df

    def append_blacklist(df: pd.DataFrame):
        """
        It append the variable FRAUDE = 1 in the dataframe passed.
        :return: df + FRAUDE
        """
        from utils.read_csv import ReadCsv
        from utils.dataframe_utils import DfUtils

        file_blacklist_resume = ReadCsv.load_csv('siniestros_bl_processed.csv')
        df_bl_resume = DfUtils.processing_file(file_blacklist_resume, delimiter=';')
        df_bl_resume['ID_SINIESTRO'] = df_bl_resume['ID_SINIESTRO'].map(int)
        df_bl_resume = df_bl_resume.drop_duplicates(subset='ID_SINIESTRO')
        df['id_siniestro'] = df['id_siniestro'].map(int)
        df = pd.merge(df, df_bl_resume, how='left', left_on='id_siniestro', right_on='ID_SINIESTRO')
        df['FRAUDE'] = pd.Series(0, index=df.index)
        df.loc[df['ID_SINIESTRO'].notnull(), 'FRAUDE'] = 1
        del df_bl_resume
        del df['ID_SINIESTRO']

        return df

    def correlation_get_column(df: pd.DataFrame, columns: list =[], output_file=False, show_plot=False):
        """
        We get correlations for specific columns selected in 'columns' list.
        :param columns: Columns of the df we want to correlate.
        :param output_file: If we want a csv with the output.
        :param show_plot: If we want a Heat Map.
        :return: df subset correlations.
        """
        import seaborn as sns
        index = []
        for i in columns:
            index_i = df.columns.get_loc(i)
            index.append(index_i)
        corrmat = df.corr().iloc[:, index]
        print(corrmat)

        if output_file:
            corrmat.to_csv('corrmat_subset.csv', sep=';')

        if show_plot:
            f, ax = plot.subplots(figsize=(12, 9))
            sns.heatmap(corrmat, vmax=.8, square=True)
            networks = corrmat.columns.values.tolist()
            for i, network in enumerate(networks):
                if i and network != networks[i - 1]:
                    ax.axhline(len(networks) - i, c="w")
                    ax.axvline(i, c="w")
            f.tight_layout()
            plot.show()

    def correlation_get_all(df: pd.DataFrame, get_all=False, get_specific='FRAUDE', output_file=False, show_plot=False):
        """
        We get correlations for a specific column or the whole dataframe.
        :param get_all: True if we want the whole dataframe correlation.
        :param get_specific: Column of the df we want to correlate.
        :param output_file: If we want a csv with the output.
        :param show_plot: If we want a Heat Map.
        :return: df correlations.
        """
        import seaborn as sns
        if get_all:
            corrmat = df.corr()
        else:
            index_i = df.columns.get_loc(get_specific)
            corrmat = df.corr().iloc[:, index_i]
        print(corrmat)

        if output_file:
            corrmat.to_csv('corrmat_whole.csv', sep=';')

        if show_plot:
            f, ax = plot.subplots(figsize=(12, 9))
            sns.heatmap(corrmat, vmax=.8, square=True)
            networks = corrmat.columns.values.tolist()
            for i, network in enumerate(networks):
                if i and network != networks[i - 1]:
                    ax.axhline(len(networks) - i, c="w")
                    ax.axvline(i, c="w")
            f.tight_layout()
            plot.show()

    def output_normal_anormal_new(df: pd.DataFrame, output_file=True, input_name_file='raw_file'):
        """
        It split the dataframe into three dataframes (normal, anormal, new) based on FRAUDE = (0,1) and New Sinister 
        Bottle. Also, if 'output_file' = True, it creates a new version of the final table.
        :param output_file: Boolean if it is necessary an output file
        :param input_name_file: The name of the output file.
        :return: Two dataframes based on normally anormally.
        """

        import STRING

        # First we separete New sinister
        new = df[df['TEST'] == 1]
        df = df[df['TEST'] == 0]

        del new['TEST']
        del df['TEST']

        df['FRAUDE'] = df['FRAUDE'].map(int)

        anomaly = df[df['FRAUDE'] == 1]
        normal = df[df['FRAUDE'] == 0]

        print('anomaly shape ', anomaly.shape)
        print('normal shape ', normal.shape)
        string_anomaly = 'anomaly_' + input_name_file
        print('output file anomaly ', string_anomaly)
        string_normal = 'normal_' + input_name_file
        print('output file normal ', string_normal)

        if output_file:
            path = 'batch_files\\'

            normal_file = path + string_normal + '.csv'
            anormal_file = path + string_anomaly + '.csv'
            new_file = path + 'new_sinister.csv'

            anomaly.to_csv(anormal_file, sep=';', index=False)
            normal.to_csv(normal_file, sep=';', index=False)
            new.to_csv(new_file, sep=';', index=False)

        return normal, anomaly, new
