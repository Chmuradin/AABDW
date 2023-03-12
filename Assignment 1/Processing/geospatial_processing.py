import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import RadiusNeighborsClassifier

from utils import brussels_zipcodes, antwerp_zipcodes
from frequency_encoding import frequency_encode

import matplotlib

matplotlib.use("TkAgg")  # required to be like this, otherwise will crash PyCharm
from matplotlib import pyplot as plt

kms_per_radian = 6371.0088


def process_zipcodes(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Adds two 1/0 columns (brussels, antwerp) indicating if property is in Brussels/Antwerp.

    If unknown/not in both then the row has 0 on both columns.

    Encodes zipcodes with frequency.

    5276 properties in Brussels.
    1024 properties in Antwerp.

    """
    # fix possible typos
    train_df = fix_zipcode_typos(train_df)
    test_df = fix_zipcode_typos(test_df)

    # impute the missing zipcodes first
    train_df = zipcode_impute(train_df, train_df)  # fix train data first
    test_df = zipcode_impute(train_df, test_df)  # then we can fix the test data

    # change zipcode from string to int
    train_df['property_zipcode'] = train_df['property_zipcode'].astype(int)
    test_df['property_zipcode'] = test_df['property_zipcode'].astype(int)

    # check in Brussels/Antwerp or no
    test_df.loc[test_df['property_zipcode'].isin(brussels_zipcodes), 'brussels'] = 1
    test_df.loc[test_df['property_zipcode'].isin(antwerp_zipcodes), 'antwerp'] = 1
    test_df['brussels'].fillna(value=0, inplace=True)
    test_df['antwerp'].fillna(value=0, inplace=True)

    # frequency encode zipcodes
    test_df = frequency_encode(train_df=train_df, df_to_encode=test_df, column='property_zipcode',
                               normalize=True, new_column_name='zipcode_freq')

    return test_df


def zipcode_impute(train_df: pd.DataFrame, test_df: pd.DataFrame,
                   radius=0.2):  # optimize later on, check optimal parameter
    eps = radius / kms_per_radian

    train_data = train_df[train_df['property_zipcode'].notna()]
    indices_to_impute = test_df['property_zipcode'].isna().index
    test_data = test_df.iloc[indices_to_impute]

    s = RadiusNeighborsClassifier(radius=eps, weights='distance', metric='haversine', outlier_label='-99')
    s.fit(np.radians(train_data.loc[:, ['property_lon', 'property_lat']]), train_data['property_zipcode'])

    predictions = s.predict(np.radians(test_data.loc[:, ['property_lon', 'property_lat']]))

    test_df.loc[indices_to_impute, 'property_zipcode'] = predictions

    return test_df


def fix_zipcode_typos(df):
    typo_indexes = df.index[df['property_zipcode'] == '11 20']  # typo
    for idx in typo_indexes:
        df.loc[idx, 'property_zipcode'] = '1120'

    return df


def cluster_long_lat(train_df: pd.DataFrame, test_df: pd.DataFrame,
                     distance=1, min_samples=3):
    """
    Idea taken from:
    https://geoffboeing.com/2014/08/clustering-to-reduce-spatial-data-set-size/
    https://github.com/gboeing/2014-summer-travels/blob/master/clustering-scikitlearn.ipynb
    """

    train_coords = train_df.loc[:, ['property_lat', 'property_lon']]
    test_coords = test_df.loc[:, ['property_lat', 'property_lon']]

    epsilon = distance / kms_per_radian

    db = DBSCAN(eps=epsilon, min_samples=min_samples, algorithm='ball_tree', metric='haversine')
    db.fit(np.radians(train_coords))

    return pd.Series(db.labels_)


def visualize_cluster(lon, lat, labels):
    num_of_labels = len(set(labels))
    colormap = plt.cm.gist_ncar
    colors = [colormap(i) for i in np.linspace(0, 0.99, num_of_labels)]
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, num_of_labels))))

    for label in range(-1, len(set(labels))):
        x = lon[labels == label]
        y = lat[labels == label]
        plt.scatter(x, y)

    plt.show()


def check_clustering(train_df: pd.DataFrame, test_df: pd.DataFrame,
                     distance=1, min_samples=3):
    labels = cluster_long_lat(train_df, test_df, distance=distance, min_samples=min_samples)
    visualize_cluster(lon=train_df['property_lon'], lat=train_df['property_lat'], labels=labels)

    num_of_labels = len(set(labels))
    print(f'There are {num_of_labels} clusters.')
