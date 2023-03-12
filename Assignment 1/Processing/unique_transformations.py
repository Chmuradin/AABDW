import pandas as pd
from math import ceil

from utils import has_word


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


def quick_fixes(df: pd.DataFrame):
    df['property_bathrooms'].fillna(value=1, inplace=True)  # just assume at least 1 bathroom
    df['property_bedrooms'].fillna(value=0, inplace=True)  # assuming 0 bedrooms (quite common)
    df['reviews_per_month'].fillna(value=0, inplace=True)  # only properties with no reviews have NAs here

    # assume 2 people per bed
    no_beds_indexes = df[df['property_beds'].isna()].index
    for idx in no_beds_indexes:
        df.loc[idx, 'property_beds'] = ceil(df.loc[idx, 'property_max_guests']/2)

    return df


def quick_numerical_impute(df: pd.DataFrame, column):  # numerical only
    value = df[column].median()
    df[column].fillna(value=value, inplace=True)

    return df






















