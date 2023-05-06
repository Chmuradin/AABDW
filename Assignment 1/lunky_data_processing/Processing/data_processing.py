from utils import read_data

from geospatial_processing import process_zipcodes
from categorical_encoding import process_categorical
from checklist_based import process_extra, process_property_amenities, process_host_verified
from ordinal_transformations import process_booking_cancel_policy, process_host_response_time
from unique_transformations import process_host_location, quick_fixes, quick_numerical_impute
from time_processing import process_host_since, process_last_updated


def transform_data(train_df_path, test_df_path,
                   freq_encoding: list, normalize_freq: bool,
                   one_hot_encoding: list,
                   numerical_impute: list):  # keep in mind which processes need a train & test data
    """

    :param normalize_freq: if True, then freq encoding, if false, count encoding
    :param one_hot_encoding: categorical to one hot encode
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

    test_df = process_categorical(train_df, test_df, freq_encoding, one_hot_encoding, normalize_freq)

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

    # test_df.drop(columns=['host_nr_listings', 'host_nr_listings_total'], inplace=True)  # irrelevant
    # 'host_nr_listings', 'host_nr_listings_total' is the exact same, and already covered by 'host_id_freq'

    # time transformations
    test_df = process_host_since(test_df)
    test_df = process_last_updated(test_df)

    return test_df


def main(train_df_path, test_df_path,
         to_freq_encode=None, normalize_freq=True,
         to_onehot_encode=None,
         to_numeric_impute=None):  # for sandbox usage, use './Data/train.csv' for both

    if type(to_freq_encode) == int:
        to_freq_encode = []
    elif not to_freq_encode:  # just to supply default values
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

    if not to_onehot_encode:
        to_onehot_encode = []

    df = transform_data(train_df_path, test_df_path, freq_encoding=to_freq_encode,
                        normalize_freq=normalize_freq,
                        one_hot_encoding=to_onehot_encode,
                        numerical_impute=to_numeric_impute)

    return df


if __name__ == '__main__':
    from utils import train_path, test_path, data_path
    from datetime import datetime

    dframe = main(train_path, test_path,
                  to_freq_encode=['host_id', 'property_type', 'property_room_type'],
                  to_onehot_encode=[],
                  to_numeric_impute=None)

    # throw out non-numeric cols for now
    non_num_cols = list(dframe.select_dtypes(include='object').columns)
    dframe.drop(columns=non_num_cols, inplace=True)
    dframe.drop(columns=['property_scraped_at', 'host_nr_listings_total'],
                inplace=True)  # why would we need this

    version = 2
    file_name = rf'\base_test_v{version}_{str(datetime.now().date())}.csv'

    dframe.to_csv(data_path + file_name,
                  index=False)
