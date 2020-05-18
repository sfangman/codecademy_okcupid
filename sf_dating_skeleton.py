import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

#Create your df here:
df = pd.read_csv('profiles.csv')
df.columns
df.head(1)
# df.sign.replace({"&rsquo;":"'"},regex=True).value_counts()
# df.sign.head(25)
df = df[df.sign.notna()]
df['z_sign'] = df.sign.apply(lambda x: str(x).split()[0])
# df.income = df.income.replace({-1:0}).fillna(0).astype('float')

fig, ax = plt.subplots(figsize=(10,7))
plt.bar(df.groupby(['z_sign']).age.count().reset_index().z_sign, df.groupby(['z_sign']).age.count().reset_index().age)
plt.xlabel('Zodiac Sign')
plt.ylabel('Count')
plt.title('Zodiac Sign Frequency')
plt.xticks(rotation=90)
plt.show()

fig, ax = plt.subplots(figsize=(10,7))
plt.hist(df[df.income>=0].income, bins=50)
plt.xlabel('Income ($)')
plt.ylabel('Frequency')
plt.title('Income Distribution')
plt.show()

##### SEX #####
df.sex.value_counts()
df['sex_code'] = df.sex.map({'m':1, 'f':0})
df.sex_code.value_counts()

#### DIET #####
df.diet.value_counts()

# Map Diets to numeric scale assuming mostly -> strictly scale
vegan_map = {"mostly vegan":1, "vegan":2, "strictly vegan":3}
anything_map = {"mostly anything":1, "anything":2, "strictly anything":3}
vegetarian_map= {"mostly vegetarian":1, "vegetarian":2, "strictly vegetarian":3}
other_map = {"mostly other":1, "other":2, "strictly other":3}
kosher_map = {"mostly kosher":1, "kosher":2, "strictly kosher":3}
halal_map = {"mostly halal":1, "halal":2, "strictly halal":3}

# Map to numeric scale using defined maps
df['diet_vegan'] = df.diet.map(vegan_map).fillna(0)
df['diet_anything'] = df.diet.map(anything_map).fillna(0)
df['diet_vegetarian'] = df.diet.map(vegetarian_map).fillna(0)
df['diet_other'] = df.diet.map(other_map).fillna(0)
df['diet_kosher'] = df.diet.map(kosher_map).fillna(0)
df['diet_halal'] = df.diet.map(halal_map).fillna(0)

# Verify maps
df.diet_anything.value_counts()

####### JOB #######
df.job.value_counts()

# Map various jobs to individual numeric flags
df['job_other'] = df.job.map({"other":1}).fillna(0)
df['job_student'] = df.job.map({"student":1}).fillna(0)
df['job_stem'] = df.job.map({"science / tech / engineering":1}).fillna(0)
df['job_tech'] = df.job.map({"computer / hardware / software":1}).fillna(0)
df['job_artist'] = df.job.map({"artistic / musical / writer":1}).fillna(0)
df['job_sales'] = df.job.map({"sales / marketing / biz dev":1}).fillna(0)
df['job_medical'] = df.job.map({"medicine / health":1}).fillna(0)
df['job_education'] = df.job.map({"education / academia":1}).fillna(0)
df['job_management'] = df.job.map({"executive / management":1}).fillna(0)
df['job_finance'] = df.job.map({"banking / financial / real estate":1}).fillna(0)
df['job_media'] = df.job.map({"entertainment / media":1}).fillna(0)
df['job_law'] = df.job.map({"law / legal services":1}).fillna(0)
df['job_travel'] = df.job.map({"hospitality / travel":1}).fillna(0)
df['job_construction'] = df.job.map({"construction / craftsmanship":1}).fillna(0)
df['job_admin'] = df.job.map({"clerical / administrative":1}).fillna(0)
df['job_gov'] = df.job.map({"political / government":1}).fillna(0)
df['job_rns'] = df.job.map({"rather not say":1}).fillna(0)
df['job_transportation'] = df.job.map({"transportation":1}).fillna(0)
df['job_unemployed'] = df.job.map({"unemployed":1}).fillna(0)
df['job_retired'] = df.job.map({"retired":1}).fillna(0)
df['job_military'] = df.job.map({"military":1}).fillna(0)

####### EDUCATION #######
df.education.value_counts()
# Map to graduation of each school
df['grad_bachelor'] = df.education.map({"graduated from college/university":1}).fillna(0)
df['grad_masters'] = df.education.map({"graduated from masters program":1}).fillna(0)
df['grad_associate'] = df.education.map({"graduated from two-year college":1}).fillna(0)
df['grad_hs'] = df.education.map({"graduated from high school":1}).fillna(0)
df['grad_law'] = df.education.map({"graduated from law school":1}).fillna(0)
df['grad_phd'] = df.education.map({"graduated from ph.d program":1}).fillna(0)
df['grad_med'] = df.education.map({"graduated from med school":1}).fillna(0)
df['grad_space'] = df.education.map({"graduated from space camp":1}).fillna(0)

# Verify
df.grad_hs.value_counts()

####### DRINKS, SMOKE, DRUGS #######
df.drugs.value_counts()
# Drinks, Smokes, Drugs Mappings
drink_mapping = {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5}
smoke_mapping= {"no":0, "when drinking":1, "trying to quit":2, "sometimes":3, "yes":4}
drug_mapping = {"never":0, "sometimes":1, "often":2}
# Apply maps
df['drinks_code'] = df.drinks.map(drink_mapping)
df['smoke_code'] = df.smokes.map(smoke_mapping)
df['drug_code'] = df.drugs.map(drug_mapping)
# Verify
df.drug_code.value_counts()

####### ESSAYS #######
essay_cols = ["essay0","essay1","essay2","essay3","essay4","essay5","essay6","essay7","essay8","essay9"]
# Removing the NaNs
all_essays = df[essay_cols].replace(np.nan, '', regex=True)
# Combining the essays
all_essays = all_essays[essay_cols].apply(lambda x: ' '.join(x), axis=1)
# Essay Length
df["essay_len"] = all_essays.apply(lambda x: len(x))
# Essay average word length
df["essay_word_len"] = all_essays.apply(lambda x: (sum(len(word) for word in x.split())/len(x.split())) if len(x.split())>0 else 0)
df.columns

print('Classification Models - Zodiac Sign')
# feature_data = df[['age','diet_vegan',
#        'diet_anything', 'diet_vegetarian', 'diet_other', 'diet_kosher',
#        'diet_halal', 'job_other', 'job_student', 'job_stem', 'job_tech',
#        'job_artist', 'job_sales', 'job_medical', 'job_education',
#        'job_management', 'job_finance', 'job_media', 'job_law', 'job_travel',
#        'job_construction', 'job_admin', 'job_gov', 'job_rns',
#        'job_transportation', 'job_unemployed', 'job_retired', 'job_military',
#        'drinks_code', 'smoke_code', 'drug_code', 'essay_len',
#        'essay_word_len', 'z_sign']]

feature_data = df[['age','diet_vegan',
       'diet_anything', 'diet_vegetarian', 'diet_other', 'diet_kosher',
       'diet_halal',
       'drinks_code', 'smoke_code', 'drug_code', 'essay_len',
       'essay_word_len', 'z_sign']]

feature_data_df = feature_data.dropna().reset_index(drop=True)

feature_data_df.head()
from sklearn.preprocessing import MinMaxScaler

features = feature_data_df.drop(['z_sign'], axis=1)

print('Model Features:')
print(list(features.columns))

x = features.values

scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)
features = pd.DataFrame(x_scaled, columns=features.columns)
features.head()

labels = feature_data_df['z_sign']

from sklearn.model_selection import train_test_split

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=1)

from sklearn.linear_model import LogisticRegression
from sklearn import metrics

lr_classifier = LogisticRegression(solver='lbfgs', multi_class='auto')
lr_classifier.fit(train_features, train_labels)
predicted_labels = lr_classifier.predict(test_features)
predicted_probabilities = lr_classifier.predict_proba(test_features)

print("Logistic Regression Classifier\naccuracy_score={}\nprecision={}\nrecall={}\nf1_score={}".format(
    metrics.accuracy_score(test_labels, predicted_labels),
    metrics.precision_score(test_labels, predicted_labels, average='micro'),
    metrics.recall_score(test_labels, predicted_labels, average='micro'),
    metrics.f1_score(test_labels, predicted_labels, average='micro')
))

coef_df = pd.DataFrame(lr_classifier.coef_, columns=features.columns, index=lr_classifier.classes_)

ax, fig = plt.subplots(figsize=(12,7))
sns.heatmap(coef_df, center=0)
plt.title('Logistic Regression Model Coefficient Heatmap')
plt.subplots_adjust(bottom=0.25)
plt.show()

from sklearn.neighbors import KNeighborsClassifier
### Loop to determine optimal k (k=19 for sake of repetition)
max_f1=0
k_opt=0
f1_scores = []
print("Looping through k = ", end='')
for k in range(1,21):
    print(str(k)+',', end='', flush=True)
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(train_features,train_labels)
    predicted_labels = knn.predict(test_features)
    f1 = metrics.f1_score(test_labels, predicted_labels, average='micro')
    f1_scores.append(f1)
    if f1 > max_f1:
        max_f1 = f1
        k_opt = k

fig, ax = plt.subplots(figsize=(10,7))
plt.plot(range(1,21), f1_scores)
plt.title('K Neighbors Classifier')
plt.xlabel('k (n_neighbors)')
plt.ylabel('f1 score')
ax.annotate('k={}, f1={}'.format(k_opt,max_f1), (k_opt, f1_scores[k_opt-1]))
plt.show()
# k_opt=19 #For repetition
knn = KNeighborsClassifier(n_neighbors = k_opt)
knn.fit(train_features,train_labels)
predicted_labels = knn.predict(test_features)
predicted_probabilities = knn.predict_proba(test_features)

print("K Neighbors Classifier (k={})\naccuracy_score={}\nprecision={}\nrecall={}\nf1_score={}".format(
    k_opt,
    metrics.accuracy_score(test_labels, predicted_labels),
    metrics.precision_score(test_labels, predicted_labels, average='micro'),
    metrics.recall_score(test_labels, predicted_labels, average='micro'),
    metrics.f1_score(test_labels, predicted_labels, average='micro')
))

from sklearn.ensemble import RandomForestClassifier
print('Random Forest Classifier')
### Loop to determine optimal d (d=9 for sake of repetition)
max_f1=0
d_opt=0
f1_scores = []
print("Looping through d = ", end='')
for d in range(1,11):
    print(str(d)+',', end='', flush=True)
    rf = RandomForestClassifier(max_depth=d, n_estimators=100, random_state=1)
    rf.fit(train_features, train_labels)
    predicted_labels = rf.predict(test_features)
    f1 = metrics.f1_score(test_labels, predicted_labels, average='micro')
    f1_scores.append(f1)
    if f1 > max_f1:
        max_f1 = f1
        d_opt = d

fig, ax = plt.subplots(figsize=(10,7))
plt.plot(range(1,11), f1_scores)
plt.title('Random Forest Classifier')
plt.xlabel('d (max_depth)')
plt.ylabel('f1 score')
ax.annotate('d={}, f1={}'.format(d_opt,max_f1), (d_opt, f1_scores[d_opt-1]))
plt.show()

# d_opt=9 #For repetition
rf = RandomForestClassifier(max_depth=d_opt, n_estimators=100, random_state=1)
rf.fit(train_features, train_labels)
predicted_labels = rf.predict(test_features)

print("Random Forest Classifier (d={})\naccuracy_score={}\nprecision={}\nrecall={}\nf1_score={}".format(
    d_opt,
    metrics.accuracy_score(test_labels, predicted_labels),
    metrics.precision_score(test_labels, predicted_labels, average='micro'),
    metrics.recall_score(test_labels, predicted_labels, average='micro'),
    metrics.f1_score(test_labels, predicted_labels, average='micro')
))

from sklearn.svm import SVC

print('Support Vector Machine Classifier')
### Loop to optimize SVM
# opt_g = 0
# opt_c = 0
# max_f1 = 0
# print("Looping through (g,c) = ", end='')
# for g in range(1,5):
#     for c in range(1,5):
#         print("({},{})".format(g,c), end='', flush=True)
#         svc = SVC(kernel='rbf', gamma=g, C=c)
#         svc.fit(train_features, train_labels)
#         predicted_labels  = svc.predict(test_features)
#         f1 = metrics.f1_score(test_labels, predicted_labels, average='micro')
#         if f1 > max_f1:
#             max_f1 = f1
#             opt_g = g
#             opt_c = c

svc = SVC(kernel='rbf', cache_size=1000)
svc.fit(train_features, train_labels)
predicted_labels  = svc.predict(test_features)

print("SVM Classifier\naccuracy_score={}\nprecision={}\nrecall={}\nf1_score={}".format(
    metrics.accuracy_score(test_labels, predicted_labels),
    metrics.precision_score(test_labels, predicted_labels, average='micro'),
    metrics.recall_score(test_labels, predicted_labels, average='micro'),
    metrics.f1_score(test_labels, predicted_labels, average='micro')
))


print('Regression Models - Income')

# New DF for incomes
df_inc = df[df.income >= 0]
df_inc.income.value_counts()
# df.corr()['income']

feature_data = df_inc[['age','sex_code','job_other', 'job_student', 'job_stem',
       'job_tech', 'job_artist', 'job_sales', 'job_medical', 'job_education',
       'job_management', 'job_finance', 'job_media', 'job_law', 'job_travel',
       'job_construction', 'job_admin', 'job_gov', 'job_rns',
       'job_transportation', 'job_unemployed', 'job_retired', 'job_military','essay_len', 'essay_word_len',
       'grad_bachelor', 'grad_masters', 'grad_associate', 'grad_hs',
       'grad_law', 'grad_phd', 'grad_space', 'grad_med','income']]

feature_data_df = feature_data.dropna().reset_index(drop=True)
feature_data_df.head()

features = feature_data_df.drop(['income'], axis=1)

print('Model Features:')
print(list(features.columns))
print('Scaling Features')

x = features.values

scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)
features = pd.DataFrame(x_scaled, columns=features.columns)
features.head()

targets = feature_data_df['income']

train_features, test_features, train_targets, test_targets = train_test_split(features, targets, test_size=0.2, random_state=1)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(train_features, train_targets)
print('Model Score (R2) = {}'.format(lr.score(test_features, test_targets)))
predicted_targets = lr.predict(test_features)
fig, ax = plt.subplots(figsize=(10,7))
plt.scatter(predicted_targets, test_targets)
plt.xlabel('Predicted Income')
plt.ylabel('Actual Income')
plt.title('Multiple Linear Regression Results')
plt.show()

lr_coefs = pd.DataFrame({'col':list(features.columns), 'coef':lr.coef_, 'abs_coef':abs(lr.coef_)}).sort_values(by='abs_coef', ascending=False).reset_index(drop=True)
lr_coefs

from sklearn.neighbors import KNeighborsRegressor

k_opt = 0
max_score = 0
scores = []
print('Iteration: k = ', end='')
for k in range(1,21):
    print('{}, '.format(k), end='', flush=True)
    knr = KNeighborsRegressor(n_neighbors=k, weights='distance')
    knr.fit(train_features, train_targets)
    score = knr.score(test_features, test_targets)
    scores.append(score)
    if abs(score) > abs(max_score):
        k_opt = k
        max_score = score

fig, ax = plt.subplots(figsize=(10,7))
plt.plot(range(1,21), scores)
plt.xlabel('n_neighbors (k)')
plt.ylabel('R2 Score')
plt.title('K Neighbors Regressor')
ax.annotate('k={}, R2={}'.format(k_opt,max_score), (k_opt, scores[k_opt-1]))
plt.show()

knr = KNeighborsRegressor(n_neighbors=k_opt, weights='distance')
knr.fit(train_features, train_targets)
print('\nModel Score (R2) = {}'.format(knr.score(test_features, test_targets)))
predicted_targets = knr.predict(test_features)
fig, ax = plt.subplots(figsize=(10,7))
plt.scatter(predicted_targets, test_targets)
plt.xlabel('Predicted Income')
plt.ylabel('Actual Income')
plt.title('K Neighbors Regression Results - k={}'.format(k_opt))
plt.show()


from sklearn.ensemble import RandomForestRegressor

d_opt = 0
max_score = 0
scores = []
print('Iteration: d = ', end='')
for d in range(1,11):
    print('{}, '.format(d), end='', flush=True)
    rfr = RandomForestRegressor(max_depth=d, n_estimators=100, random_state=1)
    rfr.fit(train_features, train_targets)
    score = rfr.score(test_features, test_targets)
    scores.append(score)
    if abs(score) > abs(max_score):
        d_opt = d
        max_score = score

fig, ax = plt.subplots(figsize=(10,7))
plt.plot(range(1,11), scores)
plt.xlabel('max depth (d)')
plt.ylabel('R2 Score')
plt.title('Random Forest Regressor')
ax.annotate('d={}, R2={}'.format(d_opt,max_score), (d_opt, scores[d_opt-1]))
plt.show()

rfr = RandomForestRegressor(max_depth=d_opt, n_estimators=100, random_state=1)
rfr.fit(train_features, train_targets)
print('\nModel Score (R2) = {}'.format(rfr.score(test_features, test_targets)))
predicted_targets = rfr.predict(test_features)
fig, ax = plt.subplots(figsize=(10,7))
plt.scatter(predicted_targets, test_targets)
plt.xlabel('Predicted Income')
plt.ylabel('Actual Income')
plt.title('Random Forest Regression Results - d={}'.format(d_opt))
plt.show()
