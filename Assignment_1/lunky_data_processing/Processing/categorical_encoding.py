import category_encoders as ce
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np


def process_categorical(train_df: pd.DataFrame, test_df: pd.DataFrame,
                        freq_encoding: list, one_hot_encoding: list, normalize_freq: bool):
    # sanity check for freq/one_hot encodings, shouldn't do both to a column
    warning_msg = f'Columns to freq and one_hot encode must be disjoint!\n Common columns are {set(freq_encoding).intersection(set(one_hot_encoding))} '
    assert (set(freq_encoding).isdisjoint(set(one_hot_encoding))), warning_msg

    # frequency encodings
    for column in freq_encoding:
        if column != 'property_bed_type':  # just bcz
            new_col_name = column + '_freq'
            test_df = frequency_encode(train_df=train_df, df_to_encode=test_df, column=column,
                                       normalize=normalize_freq, new_column_name=new_col_name)
        else:
            test_df = frequency_encode(train_df=train_df, df_to_encode=test_df, column='property_bed_type',
                                       normalize=normalize_freq, new_column_name='bed_type_freq',
                                       min_proportion=0.05)  # ensures only real-bed vs others

    # one hot encodings
    for column in one_hot_encoding:
        test_df = one_hot_encode(train_df, test_df, column)

    return test_df


def frequency_encode(train_df: pd.DataFrame, df_to_encode: pd.DataFrame, column,
                     normalize=False, new_column_name=None, min_proportion=0.01):
    """

    :param min_proportion: categories with less than 1% frequency will be categorized as other
    :param train_df: We need to train the encoding on a dataframe.
    :param df_to_encode: This is the dataframe we will transform later. Think test_data.
    :param column: The column to encode.
    :param normalize: If True then returns frequencies, otherwise return counts.
    :param new_column_name: If not False, then replaces the name of the encoded column.
    :return: Dataframe with chosen column replaced with frequency encoding. Inplace=True.
    """
    encoder = ce.CountEncoder(cols=column, normalize=normalize, min_group_size=min_proportion)
    encoder.fit_transform(train_df[column])

    df_to_encode[column] = encoder.transform(df_to_encode[column])
    if new_column_name:
        df_to_encode.rename(columns={column: new_column_name}, inplace=True)

    return df_to_encode


def one_hot_encode(train_df: pd.DataFrame, df_to_encode: pd.DataFrame, column,
                   min_frequency=None, max_categories=None):
    one_hot = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='infrequent_if_exist',
                            min_frequency=0.01, max_categories=max_categories)

    train_categorical = np.array(train_df[column]).reshape(-1, 1)
    to_encode_categorical = np.array(df_to_encode[column]).reshape(-1, 1)

    one_hot.fit(train_categorical)
    one_hot_matrix = one_hot.transform(to_encode_categorical)

    feature_names_raw = list(one_hot.get_feature_names_out())
    feature_names = [column + '_' + category.lstrip('x0_') for category in feature_names_raw]

    return drop_and_replace_one_hot(df_to_encode, column, one_hot_matrix, feature_names)


def drop_and_replace_one_hot(df, column, one_hot_matrix: np.ndarray, feature_names):
    one_hot_df = pd.DataFrame(data=one_hot_matrix, columns=feature_names)
    temp_drop = df.drop(columns=column)  # drop this, since already one-hot encoded

    return pd.concat([temp_drop, one_hot_df], axis=1)
