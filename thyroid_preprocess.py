
# coding: utf-8

# In[132]:


### changes NOT only appended, within previous work as well ###


# # Thyroid dataset

# Disclaimer: some decisions in this notebook were made just to enhance data handling and may be not optimal, considering both time complexity and memory requirements

# In[133]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import Imputer, LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import sklearn.metrics as metrics
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns


# In[134]:


import sys
from IPython.display import display
import itertools


# Prepare script execution for test set

# In[135]:


file_name_base = 'train' # default
MODE_SCRIPT = len(sys.argv) == 3 and sys.argv[1]=='-script'

if MODE_SCRIPT:
    # script execution
    # usage:
    # -script <file_name_base>
    file_name_base = sys.argv[2]
else:
    # run in jupyter notebook
    get_ipython().run_line_magic('matplotlib', 'inline')


# ### Utilities

# In[136]:


def percent(number, prec=2):
    return '{} %'.format(round(number*100, prec))


# In[137]:


# filter rows with all unknown columns
def all_nan_cols(data_frame):
    return data_frame[data_frame.isnull().all(axis=1)]

# filter rows with at least 1 unknown column
def any_nan_cols(data_frame):
    return data_frame[data_frame.isnull().any(axis=1)]

# filter rows with no unknown columns
def no_nan_cols(data_frame):
    return data_frame[data_frame.notnull().all(axis=1)]


# In[138]:


# list column names of specific type from dataframe
def getcols(data_frame, compare_type):
    return [col for col in data_frame if data_frame[col].dtype==compare_type]

# string columns
def scols(data_frame):
    return getcols(data_frame, compare_type=object)

# boolean columns
def bcols(data_frame):
    return getcols(data_frame, compare_type=bool)

# integer columns
def icols(data_frame):
    return getcols(data_frame, compare_type=np.int64)

# decimal columns
def fcols(data_frame):
    return getcols(data_frame, compare_type=np.float64)

# categorical columns
def catcols(data_frame):
    return scols(data_frame) + bcols(data_frame)

# numeric columns
def numcols(data_frame):
    return icols(data_frame) + fcols(data_frame)


# In[139]:


# Modified imputer allows substitution with mode, which can be used for categorical variables
# https://stackoverflow.com/a/45228286/5148218
class CatImputer(Imputer):
    def __init__(self, **kwargs):
        Imputer.__init__(self, **kwargs)

    def fit(self, X, y=None):
        # modify most_frequent strategy to be able to process categorical
        if self.strategy == 'most_frequent':
            self.fills = pd.DataFrame(X).mode(axis=0).squeeze()
            self.statistics_ = self.fills.values
            return self
        else:
            return Imputer.fit(self, X, y=y)

    def transform(self, X):
        if hasattr(self, 'fills'):
            return pd.DataFrame(X).fillna(self.fills).values.astype(str)
        else:
            return Imputer.transform(self, X)


# ### Plotting toolbox

# In[140]:


# bar chart
def bar(series, title=None):
    plt.figure()
    ax = series.value_counts().plot(kind='bar', title=title)
    # https://stackoverflow.com/questions/25447700/annotate-bars-with-values-on-pandas-bar-plots
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
    plt.show()
    plt.close()
    
# histogram
def hist(series, title=None, bins=20):
    plt.figure()
    series.plot(kind='hist', bins=bins, title=title)
    plt.show()
    plt.close()
    
# box plot
def box(series, title=None):
    plt.figure()
    series.plot(kind='box', title=title)
    plt.show()
    plt.close()

# violin plot
def violin(series):
    plt.figure()
    sns.violinplot(series)
    plt.show()
    plt.close()
    
# series of plots for continuous variables
def contplot(series, bins=20):
    hist(series, title=series.name, bins=bins)
    box(series)
    violin(series)

# scatter plot
def scatter(series_x, series_y, color_series=None):
    plt.figure(figsize=[8,8])
    plt.scatter(series_x, series_y, s=7, alpha=.5, label=None, c=color_series)
    plt.xlabel(series_x.name)
    plt.ylabel(series_y.name)
    plt.show()
    plt.close()

# q-q plot
def qqplot(series, title=None):
    plt.figure(figsize=[5,5])
    stats.probplot(series, plot=plt); # default is normal distribution
    plt.title(title)
    plt.show()
    plt.close()


# ### IO CSV

# In[141]:


def load_csv(path):
    print('Load data from {}'.format(path))
    return pd.read_csv(path, index_col=0)

def save_csv(frame, path):
    print('Save data to {}'.format(path))
    frame.to_csv(path)


# ### Paths

# In[142]:


FILE_PATH = './{}.csv'
FILE_PATH_POLISHED = './{}_polished.csv'
FILE_PATH_GLOBAL = './{}_global.csv'
FILE_PATH_GLOBAL_REMOVAL = './{}_global_removal.csv'
FILE_PATH_GLOBAL_PERCENTILE = './{}_global_percentile.csv'
FILE_PATH_KNN = './{}_knn.csv'
FILE_PATH_KNN_REMOVAL = './{}_knn_removal.csv'
FILE_PATH_KNN_PERCENTILE = './{}_knn_percentile.csv'


# ## Polish

# First look at data

# In[143]:


df = load_csv(FILE_PATH.format(file_name_base))
pd.set_option('display.max_columns', 500) # ensure display all columns
pd.set_option('display.max_colwidth', -1) # ensure display full values
display(df.head(3))


# Some columns can be divided into more

# In[144]:


df = pd.concat([df.drop(['medical_info'], axis=1), df['medical_info'].apply(eval).apply(pd.Series)], axis=1)


# In[145]:


display(df[df['class'].apply(lambda x: len(x.split('.')) != 2)])


# Every element cointains dot, so that we can split on it

# In[146]:


df[['class_sick','class_num']] = pd.DataFrame(df['class'].apply(lambda x: x.split('.')).values.tolist(), index=df.index)
df.drop(['class'], axis=1, inplace=True)
[print(col, df[col].unique()) for col in ['class_sick','class_num']];


# We do not need separator now, let's just see now how to format prediction later

# In[147]:


display(df['class_num'].apply(lambda x: x[0]).value_counts())


# In[148]:


df['class_num'] = df['class_num'].apply(lambda x: x.replace('|','').replace('_',''))
display(df.head())


# ### Nonsense detection

# Now, for the first and the last time, we browse data manually

# In[149]:


for col in df:
    print(col)
    print(df[col].dtype)
    print(df[col].unique())
    print()


# TBG contains no information

# In[150]:


df.drop(['TBG measured','TBG'], axis=1, inplace=True)


# ### Date formatting

# We need all strings for string methods

# In[151]:


print(df['date_of_birth'].apply(type).unique())


# In[152]:


display(df[df['date_of_birth'].apply(lambda x: isinstance(x,float))]['date_of_birth'])


# Replace by string temporarily

# In[153]:


df['date_of_birth'].replace(to_replace=np.nan, value=(10*'?'), inplace=True)


# See length anomalies...

# In[154]:


print(df['date_of_birth'].apply(len).unique())


# ... and non-dash separators...

# In[155]:


display(df[df['date_of_birth'].apply(lambda x: ('-' in x))]['date_of_birth'].head())
display(df[df['date_of_birth'].apply(lambda x: (not '-' in x))]['date_of_birth'].head())


# ... and their combinations

# In[156]:


display(df[df['date_of_birth'].apply(lambda x: ('-' in x) and len(x)!=10)]['date_of_birth'].head())
display(df[df['date_of_birth'].apply(lambda x: (not '-' in x) and len(x)!=10)]['date_of_birth'].head())


# For longer format, make sure time is always at the end

# In[157]:


display(df[df['date_of_birth'].apply(lambda x: len(x)==19 and x[-2:]!='00')]['date_of_birth'])


# It is, so we can convert datetime to date

# In[158]:


df['date_of_birth'] = df['date_of_birth'].apply(lambda x: x[:10])


# For shorter format, make sure year is never at the end

# In[159]:


display(df[df['date_of_birth'].apply(lambda x: len(x)==8 and int(x[-2:])>31)]['date_of_birth'])


# Is is not, so we can add missing century

# In[160]:


df['date_of_birth'] = df['date_of_birth'].apply(lambda x: '19{}'.format(x) if len(x)==8 else x)


# Now we should have a single length

# In[161]:


display(df['date_of_birth'].apply(len).unique())


# We want dash as separator

# In[162]:


df['date_of_birth'] = df['date_of_birth'].apply(lambda x: x.replace('/','-'))


# In[163]:


display(df[df['date_of_birth'].apply(lambda x: (not '-' in x))]['date_of_birth'].head())


# Make sure date components are all separated by dash into day, month, year, respectively

# In[164]:


df['date_of_birth'] = df['date_of_birth'].apply(lambda x: x.split('-'))
display(df[df['date_of_birth'].apply(lambda x: len(x)!=3)]['date_of_birth'].head())


# In[165]:


display(df[df['date_of_birth'].apply(lambda x: len(x[0])!=4)]['date_of_birth'].head())


# Always 3 components but some of them reversed

# In[166]:


df['date_of_birth'] = df['date_of_birth'].apply(lambda x: x[::-1] if len(x[0])!=4 else x)
display(df[df['date_of_birth'].apply(lambda x: len(x[0])!=4)]['date_of_birth'].head())


# Make sure month in the middle

# In[167]:


display(df[df['date_of_birth'].apply(lambda x: len(x)!=1 and int(x[1])>12)]['date_of_birth'])


# Now correct missing date

# In[168]:


df['date_of_birth'] = df['date_of_birth'].apply(lambda x: [np.nan,np.nan,np.nan] if len(x)==1 else x)


# Divide date into separate columns

# In[169]:


df[['year','month','day']] = pd.DataFrame(df['date_of_birth'].values.tolist(), index=df.index, dtype=np.float64)
df.drop(['date_of_birth'], axis=1, inplace=True)


# ### Bool columns

# In[170]:


df['sex'].replace(to_replace='F', value='female', inplace=True) # prevent changing sex's F afterwards
df['sex'].replace(to_replace='M', value='male', inplace=True)
df.replace(to_replace=['f','F','FALSE','negative'], value=False, inplace=True)
df.replace(to_replace=['t','T','TRUE','sick'], value=True, inplace=True)


# ### ... measured columns

# Although the information whether measured or not can be obtained from numeric columns, we keep corresponding bool ones, because the fact that patient needed to have it measured can imply disease. Note here that a situation may occur when we know a substance has been measured but its value is unknown or vice versa

# In[171]:


print(df[df['FTI measured']=='??']['FTI'].unique())
print(df[pd.isnull(df['FTI measured'])]['FTI'].unique())


# Unknown boolean means measured for FTI

# In[172]:


df['FTI measured'].replace(to_replace=['??',np.nan], value=True, inplace=True)
df['FTI'].replace(to_replace='?', value=np.nan, inplace=True)


# Make sure no other anomalies

# In[173]:


for col_measured in df.filter(regex='.measured'):
    col = col_measured.split()[0]
    display(df[df[col].apply(lambda x: type(x)==np.float64)])


# ### Data types

# Goal is to have types:
# - object for string
# - boolean (sometimes 3rd option is necessary for unknown, thn we handle like previous)
# - numPy's float64 for all numbers (to take advantage of NaN)

# In[174]:


df.info()


# Convert integers to float

# In[175]:


[print(v) for v in df['class_num'] if type(v)!=str];


# In[176]:


for col in ['FTI','class_num']+icols(df):
    df[col] = df[col].astype(np.float64)


# From this point, every integer in the frame is also represented by float, so that unknown value can be NaN

# ### Strings

# Remove spaces

# In[177]:


for col in scols(df):
    print(col)
    df[col] = df[col].apply(lambda x: x.replace(' ',''))


# For workclass column we have observed inconsistency

# In[178]:


print(df['workclass'].unique())


# In[179]:


df['workclass'] = df['workclass'].apply(lambda x: x.lower())
print(df['workclass'].unique())


# All unknown strings are going to be represented temporarily by question mark

# In[180]:


df[scols(df)] = df[scols(df)].applymap(lambda x: '?' if pd.isnull(x) else str(x))
for col in scols(df):
    print(col)
    print(df[col].unique())


# Control check on types

# In[181]:


df.info()


# ## Explore

# ### Categorical

# In[182]:


[bar(df[col], title=col) for col in catcols(df)];


# Some columns can be generalized

# In[183]:


df['workclass_basic'] = df['workclass'].apply(pd.Series)
df['workclass_basic'].replace(regex='.*gov', value='gov', inplace=True)
df['workclass_basic'].replace(regex='self-emp.*', value='self-emp', inplace=True)
for col in ['workclass','workclass_basic']:
    bar(df[col], title='workclass: {} distinct'.format(df[col].nunique()))


# The problem is that in this case similar values do not necessarily imply the same disease risk, or even worse there are not enough value counts to find it out

# In[184]:


for v in df['workclass'].unique():
    temp = df[df['workclass']==v]
    bar(temp['class_sick'], title='{} {}'.format(v, percent(len(temp[temp['class_sick']==True]) / len(temp))))


# In[185]:


df['university'] = df['education'].apply(lambda x: x.lower()).apply(pd.Series)

df['university'].replace(to_replace='preschool', value='Before age', inplace=True)
df['university'].replace(regex='.*th', value='Before age', inplace=True)

df['university'].replace(to_replace='hs-grad', value='No', inplace=True)

df['university'].replace(to_replace=['some-college','bachelors'], value='Yes', inplace=True)
df['university'].replace(regex='assoc.*', value='Yes', inplace=True)

df['university'].replace(to_replace=['masters','doctorate'], value='Yes, advanced', inplace=True)
df['university'].replace(regex='prof.*', value='Yes, advanced', inplace=True)

for col in ['education','university']:
    bar(df[col], title='{}: {} distinct'.format(col, df[col].nunique()))


# In[186]:


df['native-country'].replace(to_replace='South', value='?', inplace=True)
df['native-country'].replace(to_replace='Hong', value='Hong-Kong', inplace=True)

df['continent'] = df['native-country'].apply(pd.Series)

continents = {}
continents.update(dict.fromkeys(['United-States','Canada','Outlying-US(Guam-USVI-etc)'],                                'North America'))
continents.update(dict.fromkeys(['Mexico','Puerto-Rico','Guatemala','Columbia','Jamaica','El-Salvador',                                 'Nicaragua','Cuba','Dominican-Republic','Haiti','Ecuador','Peru','Honduras'],                                'Latin America'))
continents.update(dict.fromkeys(['Germany','Poland','Italy','France','England','Greece','Yugoslavia','Portugal'],                                'Europe'))
continents.update(dict.fromkeys(['Trinadad&Tobago','Cambodia'],                                'Africa'))
continents.update(dict.fromkeys(['Philippines','Taiwan','China','Hong-Kong','Japan','India','Vietnam','Laos'],                                'Asia'))
##########
continents.update(dict.fromkeys(['Scotland','Ireland'],'Europe'))
continents.update(dict.fromkeys(['Iran'],'Asia'))
##########

df['continent'] = df['continent'].apply(lambda x: continents[x] if x in continents.keys() and x!='?' else x)

df['USA'] = df['native-country']=='United-States'

for col in ['native-country','continent','USA']:
    bar(df[col], title='{}: {} distinct'.format(col, df[col].nunique()))


# In[187]:


display(df[catcols(df)].describe())


# ### Numeric

# In[188]:


display(df[numcols(df)].describe(percentiles=[.001,.005,.01,.05,0.1,.25,.5,.75,.9,.95,.99,.995,.999]))


# In[189]:


[contplot(df[col], bins=50) for col in numcols(df)];


# ### Outliers

# We saw different types of outliers, that is why we choose different strategies:
# - *values against common sense:* such values can be labelled as unknown without information loss
# - *marginal values:* significantly different but despite possibly real, non-standard individuals, on those outlier substitution techniques can be applied

# In[190]:


ACCEPTED_AGE_CEIL = 120
df['age'] = df['age'].apply(lambda x: np.nan if x>ACCEPTED_AGE_CEIL or pd.isnull(x) else x)
display(df[pd.isnull(df['age'])])
contplot(df['age'])


# Before removal of (seemingly) measurement instrument's mistake, it is still good to make sure it does not represent an extremely sick patient

# In[191]:


# https://www.healthline.com/health/tsh
ACCEPTED_TSH_CEIL = 300
display(df[df['TSH']>ACCEPTED_TSH_CEIL][['TSH','class_sick']])
df['TSH'] = df['TSH'].apply(lambda x: np.nan if x>ACCEPTED_TSH_CEIL else x)
contplot(df['TSH'])


# Transform exponentially decreasing *TSH* by logarithm

# In[192]:


df['TSH_log'] = df['TSH'].apply(lambda x: x if pd.isnull(x) else np.log10(x))
contplot(df['TSH_log'])


# In[193]:


# https://www.healthline.com/health/t3
ACCEPTED_T3_CEIL = 8
display(df[df['T3']>ACCEPTED_T3_CEIL][['T3','class_sick']])
df['T3'] = df['T3'].apply(lambda x: np.nan if x>ACCEPTED_T3_CEIL else x)
contplot(df['T3'])


# In[194]:


# https://www.healthline.com/health/t4
# TT4 = total T4
# T4U = ???
ACCEPTED_TT4_CEIL = 350
display(df[df['TT4']>ACCEPTED_TT4_CEIL][['TT4','class_sick']])
df['TT4'] = df['TT4'].apply(lambda x: np.nan if x>ACCEPTED_TT4_CEIL else x)
contplot(df['TT4'])


# For some numeric columns (... measured in particular), a new boolean column could be generated based on increased/decreased value detected in individual's body (ideally with treshold estimated by a specialist). However, in this case few similar conclusion have already been made, see hyper- and hypothyroid etc.

# *fnlwgt*, i.e. sample weight tells what proportion of population the sampled individual represents

# In[195]:


df.rename(columns={'fnlwgt': 'sample-weight'}, copy=False, inplace=True)


# In[196]:


display(df[df['capital-gain']>30000][['capital-gain','class_sick']])
df['capital-gain'] = df['capital-gain'].apply(lambda x: np.nan if x==99999.0 else x)
contplot(df['capital-gain'])


# In[197]:


df['capital-loss_iszero'] = df['capital-loss'].apply(lambda x: '?' if pd.isnull(x) else x==0.0).apply(pd.Series)
bar(df['capital-loss_iszero'], title='capital-loss == 0.0')
df['capital-loss_nonzero'] = df['capital-loss'].apply(lambda x: np.nan if pd.isnull(x) or x==0.0 else x).apply(pd.Series)
contplot(df['capital-loss_nonzero'])


# In[198]:


ACCEPTED_YEAR_FLOOR = 1880
df['year'] = df['year'].apply(lambda x: np.nan if x<ACCEPTED_YEAR_FLOOR or pd.isnull(x) else x)
display(df[pd.isnull(df['year'])])
contplot(df['year'])
display(df[pd.isnull(df['age'])])


# ### Missing values

# In[199]:


[bar(df[col].apply(lambda x: 'NaN' if pd.isnull(x) else 'value'), title=col) for col in numcols(df)];


# Creating new column has been unfortunately a fail, due to too little known values

# In[200]:


df.drop(['capital-loss_nonzero'], axis=1, inplace=True)


# In[201]:


bar(df.applymap(lambda x: np.nan if type(x)==str and x=='?' else x).isnull().sum(axis=1), title='missing values count per row (i.e. per patient)')


# ### Bivariate analysis

# Scatter all pairs of variables, differentiate between <span style="color:red"><b>sick</b></span> and <span style="color:blue"><b>negative</b></span> patients.
# 
# Some variables (with exponential-like distribution in particular) can be beter explored after logarithmic-like transformation applied

# In[202]:


bool_color = df['class_sick'].apply(lambda x: 'red' if x else 'blue')
[scatter(df[colx], df[coly], color_series=bool_color) for colx,coly in itertools.combinations(numcols(df), r=2)];


# Correct observed *age*-*year* nonsense

# In[203]:


print((df['year'] + df['age']).unique())


# In[204]:


nonsense_age = df[(df['year'] + df['age']).apply(lambda x: x not in [2017,2018])][['age','year']]
display(nonsense_age)


# In[205]:


df.loc[nonsense_age.index,'year'] = np.nan
scatter(df['age'], df['year'], color_series=bool_color)


# We can see *age* & *year* represent the same information.
# Also, day & month seem improbable to be disease decoupled (in case we observed sick patients associated with a specific day or month on scatter plot we would consider to keep them)

# In[206]:


df.drop(['day','month','year'], axis=1, inplace=True)


# Confirm correlation *FTI*-*TT4* by all relevant coefficients

# In[207]:


for method in ['pearson','kendall','spearman']:
    print('{} correlation coefficient'.format(method))
    display(df[numcols(df)].corr(method)[df[numcols(df)].corr(method).applymap(lambda x: np.absolute(x) > .3)])


# We did not observe any apparent correlation apart from mentioned one

# We have noticed that *education-num* depends directly on *education* (it is discrete not continuous), some malformed occurences exist though

# In[208]:


display(df[['education','education-num']].drop_duplicates())


# Indentify concrete 1:1 relationship between education columns

# In[209]:


for edu_cat in df['education'].unique():
    print(edu_cat)
    print(df[df['education']==edu_cat]['education-num'].unique())


# In[210]:


def education_to_num(edu_cat):
    return {
        'Preschool': 1.,
        '1st-4th': 2.,
        '5th-6th': 3.,
        '7th-8th': 4.,
        '9th': 5.,
        '10th': 6.,
        '11th': 7.,
        '12th': 8.,
        'HS-grad': 9.,
        'Some-college': 10.,
        'Assoc-voc': 11.,
        'Assoc-acdm': 12.,
        'Bachelors': 13.,
        'Masters': 14.,
        'Prof-school': 15.,
        'Doctorate': 16.
    }.get(edu_cat, np.nan) # unexpected to reach default   


# The advantage of numeric over categoric form is that it defines range, hence we do not proceed with column dropping, but changing (correcting) numeric series based on categoric. Great is that *education* contains no unknown (NaN) values

# In[211]:


df['education-num'] = df['education'].apply(lambda x: education_to_num(x))
display(df[['education','education-num']].drop_duplicates())
df.drop(['education'], axis=1, inplace=True)


# Outcomes (from scatter) with focus on predicted variable:
# - low *T3* levels definitely belong to disease endangered group
# - low *T4U* levels probably belong to disease endangered group
# - disease risk grows with *age*
# - *FTI* correlates with *TT4*. Interesting observation here is that sick patients have ratio FTI/TT4 > approx. 1
# - *class_num* is uniformly distributed, probably an index

# ### Distribution

# Compare quantiles with normal distribution

# In[212]:


[qqplot(df[col], title=col) for col in numcols(df)];


# Hereby we have confirmed visually correctness of logarithm-like transformation on *TSH*

# In[213]:


df.drop(['TSH'], axis=1, inplace=True)
# keep log transformed only


# ### Relationship to predicted variable

# In[214]:


# for each attribute
for col in catcols(df):
    print(col)
    # for each unique value
    for v in df[col].unique():
        # suppose this value (and NO another value) predicts True for sick patient), evaluate
        print('\t', v, len(df[df[col] == v]))
#         tp = len(df[(df[col] == v) & (df['class_sick'] == True)])
#         fp = len(df[(df[col] == v) & (df['class_sick'] != True)])
#         fn = len(df[(df[col] != v) & (df['class_sick'] == True)])
        y_col = df[col].apply(lambda x: x==v)
        # accuracy = (tp + tn) / total 
        print(2*'\t', 'accuracy ', percent(metrics.accuracy_score(y_true=df['class_sick'], y_pred=y_col)))
        print(2*'\t', ' weighted', percent(metrics.accuracy_score(y_true=df['class_sick'], y_pred=y_col, sample_weight=df['sample-weight'])))
        # precision = tp / (tp + fp)
        print(2*'\t', 'precision', percent(metrics.precision_score(y_true=df['class_sick'], y_pred=y_col)))
        print(2*'\t', ' weighted', percent(metrics.precision_score(y_true=df['class_sick'], y_pred=y_col, sample_weight=df['sample-weight'])))
        # recall = tp / (tp + fn)
        print(2*'\t', 'recall   ', percent(metrics.recall_score(y_true=df['class_sick'], y_pred=y_col)))
        print(2*'\t', ' weighted', percent(metrics.recall_score(y_true=df['class_sick'], y_pred=y_col, sample_weight=df['sample-weight'])))
        # f1score = 2 / (1/precision + 1/recall)
        print(2*'\t', 'F1 score ', percent(metrics.f1_score(y_true=df['class_sick'], y_pred=y_col)))
        print(2*'\t', ' weighted', percent(metrics.f1_score(y_true=df['class_sick'], y_pred=y_col, sample_weight=df['sample-weight'])))


# *native-country* contains too little sample per most of countries. Besides, possible occurence of a new country is gonna be a problem for transformers

# In[215]:


df.drop(['native-country'], axis=1, inplace=True)


# Outcomes with focus on predicted variable:
# - 

# ### Inconsistency

# Inconsistencies that have occured with test set:
# - column *native-country* contains other countries which makes newly created *continent* useless since we have defined key-value dictionary manually

# We fixed it by adding additional countries to train better model, even though it may decrease its robustness

# In[216]:


##########
# continents.update(dict.fromkeys(['Scotland','Ireland'],'Europe'))
# continents.update(dict.fromkeys(['Iran'],'Asia'))
##########


# ### Finish

# Some columns are not so relevant since we are to use decision trees for classification

# In[217]:


df.drop(['sample-weight'], axis=1, inplace=True)


# In[218]:


df.replace(to_replace='?', value=np.nan, inplace=True) # because of Imputer later


# Save polished but not preserving all information, some additional information added, missing values & outliers still remain

# In[219]:


display(df.head())


# In[220]:


save_csv(df, FILE_PATH_POLISHED.format(file_name_base))


