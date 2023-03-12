from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

from utils import read_data


def apply_pca(train_df: pd.DataFrame, test_df: pd.DataFrame):  # no scree-plot, might do it later
    if 'target' in train_df:
        train_df.drop(columns='target', inplace=True)
    if 'target' in test_df:
        test_df.drop(columns='target', inplace=True)

    # scale the data first
    scaler = StandardScaler()
    scaled_train_df = pd.DataFrame(scaler.fit_transform(train_df), columns=train_df.columns)
    scaled_test_df = pd.DataFrame(scaler.transform(test_df), columns=test_df.columns)

    no_of_columns = train_df.shape[1]
    pca = PCA(n_components=no_of_columns, copy=False)

    pca.fit(scaled_train_df)

    # get at least 80% of total variance
    variance_ratios = pca.explained_variance_ratio_
    idx, components, total_variance = 0, 0, 0
    while total_variance < 0.8:
        components += 1
        total_variance += variance_ratios[idx]
        idx += 1

    transformed_test_df = pd.DataFrame(pca.transform(scaled_test_df), columns=pca.get_feature_names_out())

    return transformed_test_df.iloc[:, 0:components]


if __name__ == '__main__':
    train_path = r'C:\Users\Lunky\Desktop\Math KULeuven\Big Data Platforms & Technologies\Assigment ' \
                 r'1\AABDW\Assignment 1\Data\temp_train_data.csv '
    test_path = r'C:\Users\Lunky\Desktop\Math KULeuven\Big Data Platforms & Technologies\Assigment 1\AABDW\Assignment ' \
                r'1\Data\temp_train_data.csv '

    train_dframe = read_data(train_path)
    test_dframe = read_data(test_path)

    post_pca_test_df = apply_pca(train_dframe, test_dframe)

    post_pca_test_df.to_csv(r'C:\Users\Lunky\Desktop\Math KULeuven\Big Data Platforms & Technologies\Assigment '
                            r'1\AABDW\Assignment 1\Data\PCA_temp_train_data.csv', index=False)












