# Bank Marketing
####################################################
# About Dataset
# Context
# Find the best strategies to improve for the next marketing campaign.
# How can the financial institution have a greater effectiveness for future marketing campaigns?
# In order to answer this, we have to analyze the last marketing campaign the bank performed and
# identify the patterns that will help us find conclusions in order to develop future strategies.
#
# Source
# [Moro et al., 2014] S. Moro, P. Cortez and P. Rita.
# A Data-Driven Approach to Predict the Success of Bank Telemarketing.
# Decision Support Systems, Elsevier, 62:22-31, June 2014
#######################################################

# GENEL BAKIS
import lazypredict
from lazypredict.Supervised import LazyClassifier
import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from xgboost import XGBRegressor
import matplotlib.pyplot as pyplot
from lightgbm import LGBMClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, auc, roc_curve
from yellowbrick.classifier import ROCAUC, ClassificationReport, ClassificationScoreVisualizer
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
colors= ['#00876c','#85b96f','#f7e382','#f19452','#d43d51']

def load():
    df = pd.read_csv('bank.csv')
    return df

df = load()
df = df.drop(labels = ['default',"loan",'contact', 'day', 'month', 'pdays', 'previous', 'poutcome', 'poutcome'], axis=1)

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)

# Değişken İsimlendirilmesi
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe: dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    cat_th: int, float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th: int, float
        kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    -------
    cat_cols: list
        Kategorik değişken listesi
    num_cols: list
        Numerik değişken listesi
    cat_but_car: list
        Kategorik görünümlü kardinal değişken listesi

    Notes
    ------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool"]]

    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].nunique() < 10 and dataframe[col].dtypes in ["int64", "float64"]]

    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > 20 and str(dataframe[col].dtypes) in ["category", "object"]]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int64", "float64"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

df.head()

# Outlier Analizi
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# Aykırı Değer Analizi ve Baskılama İşlemi
for col in num_cols:
    print(col, check_outlier(df, col))

for col in num_cols:
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))

# Korelasyon Analizi
corr = df[num_cols].corr()
corr
columns = ['age', 'balance', 'duration', 'campaign']
correlation = df[columns].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation, cmap=colors, annot=True, linewidths=0.5)

plt.title('Correlation Heatmap')
plt.show()

# Kategorik Değişken Analizi
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.xticks(rotation=90)
        plt.show(block=True)


for col in cat_cols:
    cat_summary(df, col, True)

# Sayısal Değişken Analizi
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=50, figsize=(9,5))
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(df, col, True)

def num_plot(dataframe, col):
    color = "#85b96f"
    plt.figure(figsize=(10, 5))
    sns.histplot(x=dataframe[col], color=color, label=col)

    # Plotting the mean age line
    mean = df[col].mean()
    plt.axvline(x=mean, color='black', linestyle="--", label=dataframe[col].mean())

    plt.legend()
    plt.title('Distribution')
    plt.show()

num_plot(df,"age")

for col in num_cols:
    num_plot(df,col)

def target_plot(dataframe,col,palette=True):
    plt.figure(figsize=(8, 5))
    plt.title("Deposit Visualization Analysis")
    sns.countplot(x=col, hue="deposit", data=df)
    plt.xticks(rotation=70)
    plt.yticks([])
    plt.legend(title='deposit?', ncol=1, fancybox=True, shadow=True)
    plt.show()

for col in cat_cols:
    target_plot(df,col)


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df, col)


# Rare Analysis
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


rare_analyser(df, "deposit", cat_cols)


# MODELLEME İçin Gereklilikler

df = df.drop(df.loc[df["job"] == "unknown"].index)
df = df.drop(df.loc[df["education"] == "unknown"].index)

# cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=10, car_th=20)

# Modelleme için One hot işlemi
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)

df.head()

# MinMaxScaler kullanıyoruz çünkü daha kullanışlı ve eksi değerler var.
scaler = MinMaxScaler()
df["balance"] = scaler.fit_transform(df[['balance']])


standart_scaler = StandardScaler()
df["age"] = standart_scaler.fit_transform(df[['age']])

df.head()


# Modelleme

X = df.drop('deposit_1', axis=1)
y = df['deposit_1']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)


lcf = LazyClassifier(predictions = True)
models, predictions = lcf.fit(X_train, X_test, y_train, y_test)
models



lgbm_model = LGBMClassifier(random_state=42)
lgbm_model.fit(X_train, y_train,verbose=False)

lgbm_params = {"learning_rate": [0.01, 0.1, 0.001],
               "n_estimators": [100, 300, 500, 1000],
               "colsample_bytree": [0.5, 0.7, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=42).fit(X, y)

cv_results = cross_validate(lgbm_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

pred = lgbm_final.predict(X_test)
accuracy = accuracy_score(y_test, pred)

print("Accuracy:", accuracy)
print(classification_report(y_test, pred))

# Yellowbrick Raporu
fig, axs = plt.subplots(1, 2, figsize=(20, 8))
plt.suptitle("Classification Reports", family='Serif', size=15, ha='center', weight='bold')

# ROC Curve
axs[0].set_title('ROC Curve')
roc_visualizer = ROCAUC(lgbm_final, classes=[0, 1], ax=axs[0])
roc_visualizer.fit(X_train, y_train)
roc_visualizer.score(X_test, y_test)

# Sınıflandırma Raporu
axs[1].set_title('Classification Report')
classification_visualizer = ClassificationReport(lgbm_final, classes=[0, 1], support=True, ax=axs[1], cmap=colors)
classification_visualizer.fit(X_train, y_train)
classification_visualizer.score(X_test, y_test)

plt.figtext(0.05, -0.05, "Observation: Logistic Regression performed well with an accuracy score of 81%",
            family='Serif', size=14, ha='left', weight='bold')

plt.tight_layout()
plt.show()














