import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

missingCounts = train.isnull().sum()
missingColumns = missingCounts[missingCounts > 0].keys()

# t_missingCounts = test.isnull().sum()
# t_missingColumns = t_missingCounts[missingCounts > 0].keys()
#
# missingColumns.difference(t_missingColumns)
# t_missingColumns.difference(missingColumns)

for c in missingColumns:
    train[c+'_z'] = train[c].isnull().astype("int")
    test[c+'_z'] = test[c].isnull().astype("int")

train[missingColumns] = train[missingColumns].fillna(method='ffill')
train[missingColumns] = train[missingColumns].fillna(method='bfill')

test[missingColumns] = test[missingColumns].fillna(method='ffill')
test[missingColumns] = test[missingColumns].fillna(method='bfill')


def get_index(splits, index):
    return splits[index]


def split_dot(df, column, split_into_parts):
    print(column)
    tmp = df[column].map(lambda y: [x for x in str(y).split(".") if x.strip()])
    for i in range(split_into_parts):
        df[column + '_' + str(i)] = tmp.map(lambda y: get_index(y, i))
    df = df.drop([column], axis=1)
    return df


# fix
test['OsBuildLab'] = test['OsBuildLab'].map(lambda x: str(x).replace('*', '.'))

versionColumns = [('EngineVersion', 4), ('AppVersion', 4), ('AvSigVersion', 4), ('OsVer', 4),
                  ('OsBuildLab', 5), ('Census_OSVersion', 4)]

for vCol in versionColumns:
    train = split_dot(train, vCol[0], vCol[1])
    test = split_dot(test, vCol[0], vCol[1])


train['OsBuildLab_5'] = train['OsBuildLab_4'].map(lambda y: [x for x in str(y).split("-") if x.strip()][1])
train['OsBuildLab_4'] = train['OsBuildLab_4'].map(lambda y: [x for x in str(y).split("-") if x.strip()][0])

test['OsBuildLab_5'] = test['OsBuildLab_4'].map(lambda y: [x for x in str(y).split("-") if x.strip()][1])
test['OsBuildLab_4'] = test['OsBuildLab_4'].map(lambda y: [x for x in str(y).split("-") if x.strip()][0])


# train.to_csv("new_train.csv", index=None)
# test.to_csv("new_test.csv", index=None)
#
# train = pd.read_csv("new_train.csv")
# test = pd.read_csv("new_test.csv")

for c in [('EngineVersion', 4), ('AppVersion', 4), ('AvSigVersion', 4), ('OsVer', 4), ('OsBuildLab', 6),
          ('Census_OSVersion', 4)]:
    for i in range(c[1]):
        column = c[0]+'_'+str(i)
        print(column)
        print(train[column].head())


for c in [('EngineVersion', 4), ('AppVersion', 4), ('AvSigVersion', 4), ('OsVer', 4), ('OsBuildLab', 6),
          ('Census_OSVersion', 4)]:
    for i in range(c[1]):
        column = c[0] + '_' + str(i)
        print(column)
        if column == 'OsBuildLab_2' or column == 'OsBuildLab_3' or column == 'AvSigVersion_1':
            continue
        else:
            train[column] = train[column].astype('int')
            test[column] = test[column].astype('int')


objectColumns = train.select_dtypes(include=["object"]).columns

for c in objectColumns.difference(['MachineIdentifier']):
    print(c)
    le = LabelEncoder()
    train[c] = le.fit_transform(train[c])
    dic = dict(zip(le.classes_, le.transform(le.classes_)))
    test[c] = test[c].map(dic).fillna(-99.0).astype(int)

train = train.drop(['MachineIdentifier'], axis=1)

# train.to_csv("new_train1.csv", index=None)
# test.to_csv("new_test1.csv", index=None)

# train = pd.read_csv("new_train1.csv")
msk = np.random.rand(len(train)) < 0.75
test = train[~msk]
train = train[msk]

forest = RandomForestClassifier(n_estimators=50, random_state=1104, min_samples_leaf=5, n_jobs=-1, bootstrap=False)
target = train['HasDetections']
train = train.drop(['HasDetections'], axis=1)
forest = forest.fit(train, target)
forest.score(test.drop(['HasDetections'], axis=1), test['HasDetections'])

probs = forest.predict_proba(test.drop(['HasDetections'], axis=1))

test = pd.read_csv("new_test1.csv")
test['HasDetections'] = forest.predict_proba(test.drop(['MachineIdentifier'], axis=1))[:, 1]

test = test[['MachineIdentifier', 'HasDetections']]
test.to_csv("results.csv", index=False)


imp = list(zip(train.columns.difference(['HasDetections']).tolist(), forest.feature_importance_.tolist()))
imp = sorted(imp, key=lambda x: -x[1])
imp = [x for x in imp if x[1] > 0.001]

selectedColumns = [x[0] for x in imp]
selectedColumns = selectedColumns + ['HasDetections']




