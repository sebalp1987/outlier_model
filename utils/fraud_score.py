import pandas as pd


def f1(df: pd.DataFrame, class_column:str, label_column:str):

    df = df.groupby([label_column, class_column]).size().reset_index(drop=False)
    df.columns = [label_column, class_column, 'subtotal']

    df_total = df.groupby([label_column], as_index=False).sum()
    df_total = df_total[[label_column, 'subtotal']]
    df_total.columns = [label_column, 'total']

    df = pd.merge(df, df_total, on=label_column, how='left')

    f1_df = df[df[class_column] == 1]

    f1_df['weight'] = pd.Series(df['subtotal'] / df['total'], index=f1_df.index)

    f1_df['weight*x'] = pd.Series(f1_df['weight']*f1_df['subtotal'], index=f1_df.index)

    f1 = sum(f1_df['weight*x'].values) / sum(f1_df['subtotal'].values)

    print('f1 value  %.4f' % f1)
    return f1, df


def f2(df: pd.DataFrame, class_column: str, label_column: str):
    df = df.groupby([label_column, class_column]).size().reset_index(drop=False)
    df.columns = [label_column, class_column, 'subtotal']

    df_total = df.groupby([label_column], as_index=False).sum()
    df_total = df_total[[label_column, 'subtotal']]
    df_total.columns = [label_column, 'total']

    df = pd.merge(df, df_total, on=label_column, how='left')


    f2_df = df[df[class_column] == 0]

    f2_df['weight'] = pd.Series(df['subtotal'] / df['total'], index=f2_df.index)

    f2_df['weight*x'] = pd.Series(f2_df['weight'] * f2_df['subtotal'], index=f2_df.index)

    f2 = sum(f2_df['weight*x'].values) / sum(f2_df['subtotal'].values)

    print('f2 value  %.4f' % f2)
    return f2


def fraud_score(df: pd.DataFrame, class_column: str, label_column: str, beta=2, treshold=0.5):

    f1_value, df_clusters = f1(df, class_column, label_column)
    f2_value = f2(df, class_column, label_column)

    df_clusters['porcentaje'] = df_clusters['subtotal'] / df_clusters['total']
    print(df_clusters)

    df_clusters = df_clusters[df_clusters[class_column] == 1]
    df_clusters[class_column + '_Cluster'] = pd.Series(0, index=df_clusters.index)
    df_clusters.loc[df_clusters['porcentaje'] > treshold, class_column + '_Cluster'] = 1
    df_clusters = df_clusters[df_clusters[class_column + '_Cluster'] == 1]
    df_clusters = df_clusters['Clusters'].unique().tolist()
    print(df_clusters)

    fraud_score_value = (1 + beta**2) * f1_value * f2_value / (f1_value + f2_value * beta**2)

    print('fraud score % .4f' % fraud_score_value)

    return f1_value, f2_value, fraud_score_value, df_clusters
