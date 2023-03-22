import pandas as pd


def dumb_parse_text(text, sep_based_on=",", strip_based_on=" "):
    words = text.split(sep=sep_based_on)
    return [word.strip(strip_based_on) for word in words]


def has_word(text, word):
    return 1 if word.lower() in text.lower() else 0


def get_unique_values_in_a_checklist(df, col_name):
    checklist = set()
    for idx in range(len(df)):
        text = dumb_parse_text(df.loc[idx, col_name])
        checklist.update(text)
    return checklist


def read_data(path):  # apply this func to unprocessed data
    df = pd.read_csv(path, parse_dates=['property_scraped_at', 'host_since'], infer_datetime_format=True)  # or insert location of the train data

    if 'property_sqfeet' in df.columns:
        df.drop(columns='property_sqfeet', inplace=True)  # too many NAs

    return df


def read_data2(path):  # apply this func to unprocessed data
    df = pd.read_csv(path)  # or insert location of the train data

    if 'property_sqfeet' in df.columns:
        df.drop(columns='property_sqfeet', inplace=True)  # too many NAs

    return df


# https://postal-codes.cybo.com/belgium/antwerp/
antwerp_zipcodes = [2000, 2018, 2020, 2030, 2040, 2050, 2060, 2100, 2140,
                    2150, 2170, 2530, 2600, 2610, 2660]

# https://postal-codes.cybo.com/belgium/brussels/#listcodes
brussels_zipcodes = list(range(1000, 1213)) + list(range(1931, 1951))

# get_unique_values_in_a_checklist(df, 'extra')
extras_checklist = {'Host Has Profile Pic': 1,
                    'Host Identity Verified': 1,
                    'Host Is Superhost': 1,
                    'Instant Bookable': 1,
                    'Is Location Exact': 1,
                    'Require Guest Phone Verification': -1,
                    'Require Guest Profile Picture': -1,
                    'no extras': 0}

host_verified_checklist = {'None': 0,
                           'amex': 1,
                           'email': 1,
                           'facebook': 1,
                           'google': 1,
                           'government_id': 1,
                           'identity_manual': 1,
                           'jumio': 1,
                           'kba': 1,
                           'linkedin': 1,
                           'manual_offline': 1,
                           'manual_online': 1,
                           'offline_government_id': 1,
                           'phone': 1,
                           'photographer': 1,
                           'reviews': 1,
                           'selfie': 1,
                           'sent_id': 1,
                           'sesame': 1,
                           'sesame_offline': 1,
                           'work_email': 1}

host_response_time_dict = {'within an hour': 4,
                           'within a few hours': 3,
                           'within a day': 2,
                           'a few days or more': 1,
                           'unknown': 0}

booking_cancel_policy_dict = {'flexible': 4,
                              'moderate': 3,
                              'strict': 2,
                              'super_strict_30': 1}

amenities_list = ['24-hour check-in',
                  'Accessible-height bed',
                  'Air conditioning',
                  'BBQ grill',
                  'Baby bath',
                  'Baby monitor',
                  'Babysitter recommendations',
                  'Bathtub',
                  'Beach essentials',
                  'Bed linens',
                  'Breakfast',
                  'Buzzer/wireless intercom',
                  'Cable TV',
                  'Carbon monoxide detector',
                  'Cat(s)',
                  'Changing table',
                  'Children’s books and toys',
                  'Children’s dinnerware',
                  'Cleaning before checkout',
                  'Coffee maker',
                  'Cooking basics',
                  'Crib',
                  'Disabled parking spot',
                  'Dishes and silverware',
                  'Dishwasher',
                  'Dog(s)',
                  'Doorman',
                  'Doorman Entry',
                  'Dryer',
                  'Elevator in building',
                  'Essentials',
                  'Ethernet connection',
                  'Extra pillows and blankets',
                  'Family/kid friendly',
                  'Fire extinguisher',
                  'Fireplace guards',
                  'Firm matress',
                  'First aid kit',
                  'Flat smooth pathway to front door',
                  'Free parking on premises',
                  'Free parking on street',
                  'Game console',
                  'Garden or backyard',
                  'Grab-rails for shower and toilet',
                  'Gym',
                  'Hair dryer',
                  'Hangers',
                  'Heating',
                  'High chair',
                  'Hot tub',
                  'Hot water',
                  'Indoor fireplace',
                  'Internet',
                  'Iron',
                  'Keypad',
                  'Kitchen',
                  'Laptop friendly workspace',
                  'Lock on bedroom door',
                  'Lockbox',
                  'Long term stays allowed',
                  'Luggage dropoff allowed',
                  'Microwave',
                  'Other pet(s)',
                  'Outlet covers',
                  'Oven',
                  'Pack ’n Play/travel crib',
                  'Path to entrance lit at night',
                  'Patio or balcony',
                  'Pets allowed',
                  'Pets live on this property',
                  'Pocket wifi',
                  'Pool',
                  'Private bathroom',
                  'Private entrance',
                  'Private living room',
                  'Refrigerator',
                  'Roll-in shower with shower bench or chair',
                  'Room-darkening shades',
                  'Safety card',
                  'Self Check-In',
                  'Shampoo',
                  'Single level home',
                  'Smartlock',
                  'Smoke detector',
                  'Smoking allowed',
                  'Stair gates',
                  'Step-free access',
                  'Stove',
                  'Suitable for events',
                  'TV',
                  'Table corner guards',
                  'Washer',
                  'Washer / Dryer',
                  'Wheelchair accessible',
                  'Wide clearance to bed',
                  'Wide clearance to shower and toilet',
                  'Wide doorway',
                  'Wide hallway clearance',
                  'Window guards',
                  'Wireless Internet',
                  'no amenities',
                  'translation missing: en.hosting_amenity_49',
                  'translation missing: en.hosting_amenity_50']
# the last 2 are questionable... really though? I'm just summing up
# how many amenities are there, sort of like a 'luxury' index. Possible need
# of re-weighing etc
amenities_checklist = {key: 1 for key in amenities_list}
amenities_checklist['no amenities'] = 14  # this is the mean value, robustness also considered

train_path = r'C:\Users\Lunky\Desktop\Math KULeuven\Big Data Platforms & Technologies\Assigment 1\AABDW\Assignment 1\Data\train.csv'
test_path = r'C:\Users\Lunky\Desktop\Math KULeuven\Big Data Platforms & Technologies\Assigment 1\AABDW\Assignment 1\Data\test.csv'
data_path = r'C:\Users\Lunky\Desktop\Math KULeuven\Big Data Platforms & Technologies\Assigment 1\AABDW\Assignment ' \
            r'1\Data'
