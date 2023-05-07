import pandas as pd

from utils import booking_cancel_policy_dict


def process_booking_cancel_policy(df: pd.DataFrame):
    df['booking_cancel_policy'] = df['booking_cancel_policy'].apply(
        lambda x: booking_cancel_policy_dict.get(x, 0))

    df.rename(columns={'booking_cancel_policy': 'cancel_policy_flexibility'})

    return df


def process_host_response_time(df: pd.DataFrame):
    return df






