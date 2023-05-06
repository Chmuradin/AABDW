import pandas as pd


def process_host_since(df: pd.DataFrame):
    def calc_how_long_have_been_host(host_since: pd.Timestamp):
        today = pd.Timestamp('2017-05-13')  # one day since newest scraped
        how_long_host = max(0, (today-host_since).days)
        how_long_host_in_months = round(how_long_host/30)

        return how_long_host_in_months

    host_since = df.apply(lambda x: calc_how_long_have_been_host(x['host_since']),
                          axis=1)

    df['host_since'] = host_since
    df.rename(columns={'host_since': 'host_for_x_months'}, inplace=True)

    return df


# Courtesy of Aliz
def process_last_updated(df: pd.DataFrame):
    col = 'property_last_updated'
    newcol = 'last_updated_in_months'
    df[newcol] = df[col]

    # convert into days first
    for i in range(0, len(df[col])):
        if df[col][i] == "never":
            df[newcol][i] = 1  # just assume today, this is the mode
        elif df[col][i] == "today":
            df[newcol][i] = 1
        elif df[col][i] == "yesterday":
            df[newcol][i] = 2
        elif df[col][i] == "a week ago" or df[col][i] == "1 week ago":
            df[newcol][i] = 7
        else:
            for j in range(1, 58):
                if df[col][i] == str(j) + " months ago":
                    df[newcol][i] = j * 30
                elif df[col][i] == str(j) + " weeks ago":
                    df[newcol][i] = j * 7
                elif df[col][i] == str(j) + " days ago":
                    df[newcol][i] = j

    # convert into months
    df[newcol] = df.apply(lambda x: round(x[newcol]/30),  # convert into months, less flexibility
                          axis=1)

    return df

















