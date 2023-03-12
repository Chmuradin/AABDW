import category_encoders as ce
import pandas as pd


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
