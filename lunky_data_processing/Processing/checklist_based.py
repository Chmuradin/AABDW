import pandas as pd

from utils import dumb_parse_text
from utils import host_verified_checklist, amenities_checklist, extras_checklist


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

