import pandas as pd

from utils import read_data

from geospatial_processing import process_zipcodes
from frequency_encoding import frequency_encode
from checklist_based import process_extra, process_property_amenities, process_host_verified
from ordinal_transformations import process_booking_cancel_policy, process_host_response_time
from unique_transformations import process_host_location, quick_fixes, quick_numerical_impute


def transform_data(train_df_path, test_df_path,
                   freq_encoding: list,
                   numerical_impute: list):  # keep in mind which processes need a train & test data
    """

    :param numerical_impute: quick and dirty imputation of numerical columns using median
    :param train_df_path: (relative) path towards the training set
    :param test_df_path: (relative) path towards the data to actually transform
    :param freq_encoding: e.g. ['host_id', 'property_type', 'property_room_type']
    :return:
    """
    train_df = read_data(train_df_path)  # './Data/train.csv'
    test_df = read_data(test_df_path)

    # quick-fixes for bedrooms, bathrooms, and beds
    train_df = quick_fixes(train_df)
    test_df = quick_fixes(test_df)

    # most-frequent impute 'host_response_rate', 'reviews_xxx'
    for column in numerical_impute:
        train_df = quick_numerical_impute(train_df, column)
        test_df = quick_numerical_impute(test_df, column)

    # frequency encodings, needs to be done rather early
    for column in freq_encoding:
        new_col_name = column + '_freq'
        test_df = frequency_encode(train_df=train_df, df_to_encode=test_df, column=column,
                                   normalize=True, new_column_name=new_col_name)
    test_df = frequency_encode(train_df=train_df, df_to_encode=test_df, column='property_bed_type',
                               normalize=True, new_column_name='bed_type_freq',
                               min_proportion=0.05)  # ensures only real-bed vs others

    # unique transformations
    test_df = process_zipcodes(train_df, test_df)
    test_df = process_host_location(test_df)  # need to refactor slightly if want to freq_encode zipcodes

    # checklist-based transformations
    test_df = process_extra(test_df)  # do read about how this works!
    test_df = process_host_verified(test_df)
    test_df = process_property_amenities(test_df)

    # ordinal transformations
    test_df = process_host_response_time(test_df)
    test_df = process_booking_cancel_policy(test_df)

    test_df.drop(columns='property_id', inplace=True)  # irrelevant data
    test_df.drop(columns=['host_nr_listings', 'host_nr_listings_total'], inplace=True)  # irrelevant
    # 'host_nr_listings', 'host_nr_listings_total' is the exact same, and already covered by

    return test_df


def main(train_df_path, test_df_path,
         to_freq_encode=None,
         to_numeric_impute=None):  # for sandbox usage, use './Data/train.csv' for both

    if not to_freq_encode:  # just to supply default values
        to_freq_encode = ['host_id', 'property_type', 'property_room_type']

    if not to_numeric_impute:  # default values
        to_numeric_impute = ['host_response_rate',
                             'reviews_rating',
                             'reviews_acc',
                             'reviews_cleanliness',
                             'reviews_checkin',
                             'reviews_communication',
                             'reviews_location',
                             'reviews_value']

    df = transform_data(train_df_path, test_df_path, freq_encoding=to_freq_encode,
                        numerical_impute=to_numeric_impute)

    return df


if __name__ == '__main__':
    train_path = r'C:\Users\Lunky\Desktop\Math KULeuven\Big Data Platforms & Technologies\Assigment ' \
                 r'1\AABDW\Assignment 1\Data\train.csv '
    test_path = r'C:\Users\Lunky\Desktop\Math KULeuven\Big Data Platforms & Technologies\Assigment 1\AABDW\Assignment ' \
                r'1\Data\test.csv '
    dframe = main(train_path, train_path)

    # throw out non-numeric cols for now
    non_num_cols = list(dframe.select_dtypes(include='object').columns)
    dframe.drop(columns=non_num_cols, inplace=True)

    dframe.to_csv(
        r'C:\Users\Lunky\Desktop\Math KULeuven\Big Data Platforms & Technologies\Assigment 1\AABDW\Assignment '
        r'1\Data\temp_test_data.csv', index=False)
