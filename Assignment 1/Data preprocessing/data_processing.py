import pandas as pd
import os
import category_encoders as ce

from utils import brussels_zipcodes, antwerp_zipcodes, has_word, dumb_parse_text
from utils import extras_checklist, booking_cancel_policy_dict, host_verified_checklist
from utils import amenities_checklist


def read_data(path):  # apply this func to unprocessed data
    df = pd.read_csv(path)  # or insert location of the train data

    df.drop(columns='property_sqfeet', inplace=True)  # too many NAs

    return df


def process_zipcodes(df: pd.DataFrame):
    """
    Adds two 1/0 columns (brussels, antwerp) indicating if property is in Brussels/Antwerp.

    If unknown/not in both then the row has 0 on both columns.

    Removes the property_zipcode column.

    5276 properties in Brussels.
    1024 properties in Antwerp.

    """
    typo_indexes = df.index[df['property_zipcode'] == '11 20']  # typo
    for idx in typo_indexes:
        df.loc[idx, 'property_zipcode'] = '1120'

    df['property_zipcode'].fillna(value=-99, inplace=True)  # fill NAs for next step
    df['property_zipcode'] = df['property_zipcode'].astype(int)  # initially zipcodes are strings
    df.loc[df['property_zipcode'].isin(brussels_zipcodes), 'brussels'] = 1
    df.loc[df['property_zipcode'].isin(antwerp_zipcodes), 'antwerp'] = 1
    df['brussels'].fillna(value=0, inplace=True)
    df['antwerp'].fillna(value=0, inplace=True)

    df.drop(columns='property_zipcode', inplace=True)

    return df


def process_host_location(df: pd.DataFrame):
    """
    Adds a 1/0 column (host_near_property) indicating if host lives in the same city as the property.
    Only for Brussels/Antwerp! Extend if needed, but this covers 6300/6495 of cases.

    Removes the host_location column.
    """
    df['host_location'].fillna(value='', inplace=True)
    df['host_location'] = df['host_location'].apply(
        lambda x: x.lower() if type(x) == str else 'no location')

    brussels_indexes = df.index[df['brussels'] == 1]
    antwerp_indexes = df.index[df['antwerp'] == 1]

    for idx in brussels_indexes:
        text = df.loc[idx, 'host_location']
        word = 'brussels'
        df.loc[idx, 'host_near_property'] = has_word(text, word)

    for idx in antwerp_indexes:
        text = df.loc[idx, 'host_location']
        word = 'antwerp'
        df.loc[idx, 'host_near_property'] = has_word(text, word)

    df['host_near_property'].fillna(value=0, inplace=True)
    df.drop(columns='host_location', inplace=True)

    return df


def process_extra(df: pd.DataFrame):  # ugly hard coded, make it more flexible if need be
    """
    Adds an integer column (extra_score).

    Removes the extra column.

    The extra column is a checklist, with values in extras_checklist in utils.py.

    Some properties in extras_checklist are positive, some are negative, and thus are scored
    accordingly. The score is defined as the sum of these properties. Do alter value in utils.py if
    needed.
    """
    df['extra'].fillna(value='no extras', inplace=True)

    for idx in range(len(df)):
        extra = df.loc[idx, 'extra']
        checklist = dumb_parse_text(extra, sep_based_on=",", strip_based_on=" ")

        # calculate score
        score = 0
        for item in checklist:
            score += extras_checklist.get(item, 0)

        df.loc[idx, 'extra_score'] = score

    df.drop(columns='extra', inplace=True)

    return df


def process_host_response_time(df: pd.DataFrame):
    return df


def process_booking_cancel_policy(df: pd.DataFrame):
    df['booking_cancel_policy'] = df['booking_cancel_policy'].apply(
        lambda x: booking_cancel_policy_dict.get(x, 0))

    df.rename(columns={'booking_cancel_policy': 'cancel_policy_flexibility'})

    return df


def process_host_verified(df: pd.DataFrame):  # same idea as process_extra, might refactor later
    df['host_verified'].fillna(value='no verification', inplace=True)

    for idx in range(len(df)):
        host_verified = df.loc[idx, 'host_verified']
        checklist = dumb_parse_text(host_verified, sep_based_on=",", strip_based_on=" ")

        # calculate score
        score = 0
        for item in checklist:
            score += host_verified_checklist.get(item, 0)

        df.loc[idx, 'host_verified_score'] = score

    df.drop(columns='host_verified', inplace=True)
    return df


def process_property_amenities(df: pd.DataFrame):  # needs to deal with the missing values
    df['property_amenities'].fillna(value='no amenities', inplace=True)

    for idx in range(len(df)):
        prop_amenities = df.loc[idx, 'property_amenities']
        checklist = dumb_parse_text(prop_amenities, sep_based_on=",", strip_based_on=" ")

        # calculate score
        score = 0
        for item in checklist:
            score += amenities_checklist.get(item, 0)

        df.loc[idx, 'amenities_score'] = score

    df.drop(columns='property_amenities', inplace=True)
    return df


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
    encoder.fit_transform(train_df)

    df_to_encode = encoder.transform(df_to_encode)
    if new_column_name:
        df_to_encode.rename(columns={column: new_column_name}, inplace=True)

    return df_to_encode


def transform_data(train_df_path, test_df_path,
                   freq_encoding: list):  # keep in mind which processes need a train & test data
    """

    :param train_df_path: (relative) path towards the training set
    :param test_df_path: (relative) path towards the data to actually transform
    :param freq_encoding: e.g. ['host_id', 'property_type', 'property_room_type']
    :return:
    """
    train_df = read_data(train_df_path)  # './Data/train.csv'
    test_df = read_data(test_df_path)

    # frequency encodings, needs to be done first!
    for column in freq_encoding:
        new_col_name = column + '_freq'
        test_df = frequency_encode(train_df=train_df, df_to_encode=test_df, column=column,
                                   normalize=True, new_column_name=new_col_name)
    test_df = frequency_encode(train_df=train_df, df_to_encode=test_df, column='property_bed_type',
                               normalize=True, new_column_name='bed_type_freq',
                               min_proportion=0.05)  # ensures only real-bed vs others

    # unique transformations
    test_df = process_zipcodes(test_df)
    test_df = process_host_location(test_df)  # need to refactor slightly if want to freq_encode zipcodes

    # checklist-based transformations
    test_df = process_extra(test_df)  # do read about how this works!
    test_df = process_host_verified(test_df)
    test_df = process_property_amenities(test_df)

    # ordinal transformations
    test_df = process_host_response_time(test_df)
    test_df = process_booking_cancel_policy(test_df)

    test_df.drop(columns='property_id', inplace=True)

    return test_df


def main(train_df_path, test_df_path,
         to_freq_encode=None):  # for sandbox usage, use './Data/train.csv' for both

    if not to_freq_encode:  # just to supply default values
        to_freq_encode = ['host_id', 'property_type', 'property_room_type']

    df = transform_data(train_df_path, test_df_path, freq_encoding=to_freq_encode)

    return df


if __name__ == '__main__':
    absolute_path = r'C:\Users\Lunky\Desktop\Math KULeuven\Big Data Platforms & Technologies\Assigment 1\AABDW\Assignment 1\Data\train.csv'
    df = main(absolute_path, absolute_path)
    1 + 1
    df.to_csv(r'C:\Users\Lunky\Desktop\Math KULeuven\Big Data Platforms & Technologies\Assigment 1\AABDW\Assignment 1\Data\temp_train_data.csv')
