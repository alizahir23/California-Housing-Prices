#!/usr/bin/env python
# coding: utf-8

# In[608]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from scipy import stats
import joblib


# In[488]:


housing = pd.read_csv('housing.csv')
print(df.head())


# In[489]:


housing.info()


# In[490]:


housing['ocean_proximity'].value_counts()


# In[491]:


housing.describe()


# In[492]:


import matplotlib.pyplot as plt


# In[493]:


housing.hist(bins = 50, figsize=(12,8))
plt.show()


# In[494]:


train_set, test_set = train_test_split(housing, test_size=0.2, random_state= 42)


# In[495]:


print(len(train_set), len(test_set))


# In[496]:


housing['income_cat'] = pd.cut(housing['median_income'], bins=[0., 1.5, 3., 4.5, 6., np.inf], labels = [1, 2, 3, 4, 5])


# In[497]:


housing['income_cat'].value_counts().sort_index().plot.bar(rot = 0, grid = True)


# In[498]:


splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42) 
strat_splits = []

for train_index, test_index in splitter.split(housing, housing["income_cat"]):
    strat_train_set_n = housing.iloc[train_index] 
    strat_test_set_n = housing.iloc[test_index] 
    strat_splits.append([strat_train_set_n, strat_test_set_n])
    
strat_train_set, strat_test_set = strat_splits[0]


# In[499]:


strat_test_set['income_cat'].value_counts().sort_index().plot.bar()
plt.show()
strat_train_set['income_cat'].value_counts().sort_index().plot.bar()
plt.show()


# In[500]:


print('strat_test_set\n',strat_test_set['income_cat'].value_counts()/len(strat_test_set))
print('strat_train_set\n',strat_train_set['income_cat'].value_counts()/len(strat_train_set))
print('original_set\n', housing['income_cat'].value_counts()/len(housing))


# In[501]:


strat_test_set = strat_test_set.drop('income_cat', axis = 1)
strat_train_set = strat_train_set.drop('income_cat', axis = 1)


# In[502]:


housing = strat_train_set.copy()


# In[503]:


housing.plot(kind='scatter', x='longitude', y='latitude', grid=True, alpha=0.2)
plt.show()


# In[504]:


housing.plot(kind='scatter', x='longitude', y='latitude', 
             s=housing['population']/50, c='median_house_value', figsize=(13,10),
             legend=True, sharex=False, cmap='jet', grid=True, alpha=0.6 
            )
plt.show()


# In[505]:


corr_matrix = housing.select_dtypes(include=[float, int]).corr()


# In[506]:


corr_matrix['median_house_value'].sort_values(ascending=False)


# In[507]:


attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
scatter_matrix(housing[attributes], figsize = (12,8))
plt.show()


# In[508]:


housing.plot(kind='scatter', y='median_house_value', x='median_income', alpha=0.1, grid = True)
plt.show()


# In[509]:


housing['room_per_house'] = housing['total_rooms']/housing['households']
housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"] 
housing["people_per_house"] = housing["population"] / housing["households"]


# In[510]:


corr_matrix = housing.select_dtypes(include=[float, int]).corr()


# In[511]:


corr_matrix['median_house_value'].sort_values(ascending=False)


# In[519]:


housing_labels = strat_train_set['median_house_value'].copy()
housing = strat_train_set.drop('median_house_value', axis=1)


# In[520]:


imputer = SimpleImputer(strategy='median')


# In[521]:


housing_num = housing.select_dtypes(include=[np.number])
imputer.fit(housing_num)


# In[522]:


imputer.statistics_


# In[523]:


housing_num.median()


# In[524]:


X = imputer.fit_transform(housing_num)


# In[525]:


housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)


# In[526]:


housing_tr.describe()


# In[527]:


housing_cat = housing[['ocean_proximity']]
housing_cat.head(8)


# In[528]:


ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)


# In[529]:


housing_cat_encoded[:8]


# In[530]:


ordinal_encoder.categories_


# In[531]:


cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)


# In[532]:


housing_cat_1hot.toarray()
cat_encoder.feature_names_in_


# In[533]:


cat_encoder.get_feature_names_out()


# In[534]:


min_max_scaler = MinMaxScaler(feature_range = (-1, 1))
housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_num)


# In[535]:


std_scaler = StandardScaler()
housing_num_std_scaled = std_scaler.fit_transform(housing_num)


# In[536]:


age_simil_35 = rbf_kernel(housing[["housing_median_age"]], [[35]], gamma=0.1)


# In[537]:


log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
log_pop = log_transformer.transform(housing[["population"]])


# In[538]:


num_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('standardize', StandardScaler()),
])


# In[539]:


housing_num_prepared = num_pipeline.fit_transform(housing_num)
housing_num_prepared[:2].round(2)


# In[540]:


df_housing_num_prepared = pd.DataFrame(
    housing_num_prepared, 
    columns = num_pipeline.get_feature_names_out(),
    index = housing_num.index
)


# In[541]:


cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"), 
    OneHotEncoder(handle_unknown="ignore")
)


# In[542]:


preprocessing = make_column_transformer(
    (num_pipeline, make_column_selector(dtype_include=np.number)),
    (cat_pipeline, make_column_selector(dtype_include=object)),
)


# In[543]:


housing_prepared = preprocessing.fit_transform(housing)


# In[544]:


df_housing_prepared = pd.DataFrame(housing_prepared, columns = preprocessing.get_feature_names_out(), index = housing.index)
df_housing_prepared.head()


# In[545]:


class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None, n_init=10):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state
        self.n_init = n_init

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state, n_init=self.n_init)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self  # always return self!

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]


# In[546]:


def column_ratio(X): 
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in): 
    return ["ratio"] # feature names out

def ratio_pipeline(): 
    return make_pipeline(
        SimpleImputer(strategy="median"), 
        FunctionTransformer(column_ratio, feature_names_out=ratio_name), 
        StandardScaler()
    )


# In[547]:


log_pipeline = make_pipeline( 
    SimpleImputer(strategy="median"), 
    FunctionTransformer(np.log, feature_names_out="one-to-one"), 
    StandardScaler()
)

cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42, n_init=10) 

default_num_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler()
)


# In[548]:


preprocessing = ColumnTransformer([
    ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]), 
    ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]), 
    ("people_per_house", ratio_pipeline(), ["population", "households"]), 
    ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population", "households", "median_income"]), 
    ("geo", cluster_simil, ["latitude", "longitude"]),
    ("cat", cat_pipeline, make_column_selector(dtype_include=object)), ],
    remainder=default_num_pipeline) # one column remaining: housing_median_age


# In[549]:


housing_prepared = preprocessing.fit_transform(housing)
df_housing_prepared = pd.DataFrame(housing_prepared, columns=preprocessing.get_feature_names_out(), index = housing.index)
df_housing_prepared.head()


# In[552]:


lin_reg = make_pipeline(preprocessing, LinearRegression())
lin_reg.fit(housing, housing_labels)


# In[557]:


housing_predictions = lin_reg.predict(housing)
print(housing_predictions[:5].round(-2))
print(housing_labels.iloc[:5].values)


# In[563]:


lin_rmse = mean_squared_error(housing_labels, housing_predictions, squared = False)
lin_rmse


# In[571]:


tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
tree_reg.fit(housing, housing_labels)


# In[576]:


housing_predictions = tree_reg.predict(housing)
tree_rmse = mean_squared_error(housing_labels, housing_predictions, squared = False)
tree_rmse


# In[581]:


tree_rmses = -cross_val_score(tree_reg, housing, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
pd.Series(tree_rmses).describe()


# In[584]:


forest_reg = make_pipeline(preprocessing, RandomForestRegressor(random_state=42))
forest_rmses = -cross_val_score(forest_reg, housing, housing_labels, scoring="neg_root_mean_squared_error", cv=10)


# In[585]:


pd.Series(forest_rmses).describe()


# In[588]:


full_pipeline = Pipeline([
("preprocessing", preprocessing),
("random_forest", RandomForestRegressor(random_state=42)),
])
param_grid = [
{'preprocessing__geo__n_clusters': [5, 8, 10], 'random_forest__max_features': [4, 6, 8]},
{'preprocessing__geo__n_clusters': [10, 15], 'random_forest__max_features': [6, 8, 10]},
]
grid_search = GridSearchCV(full_pipeline, param_grid, cv=3, scoring='neg_root_mean_squared_error') 
grid_search.fit(housing, housing_labels)


# In[589]:


grid_search.best_params_


# In[594]:


cv_res = pd.DataFrame(grid_search.cv_results_)
cv_res.sort_values(by="mean_test_score", ascending=False, inplace=True)
cv_res[["split0_test_score", "split1_test_score", "split2_test_score", "mean_test_score"]] *= -1
cv_res.head()


# In[597]:


param_distribs = {'preprocessing__geo__n_clusters': randint(low=3, high=50), 'random_forest__max_features': randint(low=2, high=20)}
rnd_search = RandomizedSearchCV(full_pipeline, param_distributions=param_distribs, n_iter=10, cv=3, scoring='neg_root_mean_squared_error', random_state=42)
rnd_search.fit(housing, housing_labels)


# In[598]:


final_model = rnd_search.best_estimator_ # includes preprocessing
feature_importances = final_model["random_forest"].feature_importances_ 
feature_importances.round(2)


# In[599]:


sorted(zip(feature_importances, final_model["preprocessing"].get_feature_names_out()), reverse=True)


# In[600]:


X_test = strat_test_set.drop('median_house_value', axis = 1)
y_test = strat_test_set['median_house_value'].copy()


# In[604]:


final_predictions = final_model.predict(X_test)


# In[605]:


final_rmse = mean_squared_error(final_predictions, y_test, squared = False)
print(final_rmse)


# In[607]:


confidence = 0.95
squared_errors = (y_test - final_predictions) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1, loc = squared_errors.mean(), scale = stats.sem(squared_errors)))


# In[609]:


joblib.dump(final_model, "california_housing_model.pkl")

