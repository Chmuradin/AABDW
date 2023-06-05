from sklearn.decomposition import PCA
import pandas as pd

train_data = pd.read_csv("data/train_featured_origin.csv")
test_data = pd.read_csv("data/test_featured_origin.csv")

pca = PCA(n_components=10)
pca.fit(train_data)
train_pca = pd.DataFrame(pca.transform(train_data))
test_pca = pd.DataFrame(pca.transform(test_data))
