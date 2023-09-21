#!/usr/bin/env python
# coding: utf-8

# CREDIT EDA
# 
# 
# 

# In[1]:


#importing the required modules
import warnings
warnings.filterwarnings("ignore")


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option("display.max_columns",None)


# In[3]:


#reading the data set/data file(.csv)
app_data = pd.read_csv("application_data.csv")


# In[4]:


app_data.head()


# In[5]:


app_data.tail(10)


# In[6]:


app_data.describe()


# In[7]:


app_data.info()


# In[8]:


pd.set_option("display.max_rows",200)
app_data.isnull().mean()*100


# In[9]:


perc = 47
threshold = int(((100-perc)/100)*app_data.shape[0]+1)
app_df=app_data.dropna(axis=1, how='any')
app_df=app_data.dropna(axis=1, thresh=threshold)
app_df.head()


# In[10]:


app_df.info()


# In[11]:


app_df.OCCUPATION_TYPE.fillna("Others",inplace=True)


# In[12]:


app_df.OCCUPATION_TYPE.value_counts(normalize=True)*100


# In[13]:


app_df.OCCUPATION_TYPE.isnull().mean()*100


# In[14]:


app_df.EXT_SOURCE_3.isnull().mean()*100


# In[91]:


app_df.EXT_source_3.value_counts(normlaize=True)*100


# In[15]:


app_df.EXT_SOURCE_3.describe()


# In[16]:


sns.boxplot(app_df.EXT_SOURCE_3)
plt.show()


# In[17]:


app_df.EXT_SOURCE_3.fillna(app_df.EXT_SOURCE_3.median(),inplace=True)


# In[18]:


app_df.EXT_SOURCE_3.isnull().mean()*100


# In[19]:


app_df.EXT_SOURCE_3.value_counts(normalize=True)*100


# In[20]:


null_col = list(app_df.columns[app_df.isna().any()])
len(null_col)


# In[21]:


app_df.isnull().mean()*100


# In[22]:


app_df.AMT_REQ_CREDIT_BUREAU_HOUR.value_counts(normalize=True)*100


# In[23]:


app_df.AMT_REQ_CREDIT_BUREAU_DAY.value_counts(normalize=True)*100


# In[24]:


Cols = ["AMT_REQ_CREDIT_BUREAU_HOUR","AMT_REQ_CREDIT_BUREAU_DAY","AMT_REQ_CREDIT_BUREAU_WEEK","AMT_REQ_CREDIT_BUREAU_MON","AMT_REQ_CREDIT_BUREAU_QRT","AMT_REQ_CREDIT_BUREAU_YEAR"]


# In[25]:


for col in Cols:
    app_df[col].fillna(app_df[col].mode()[0],inplace=True)


# In[26]:


app_df.isnull().mean()*100


# In[27]:


null_cols = list(app_df.columns[app_df.isna().any()])
len(null_cols)


# In[28]:


app_df.NAME_TYPE_SUITE.value_counts(normalize=True)*100


# In[29]:


app_df.EXT_SOURCE_2.value_counts(normalize=True)*100


# In[30]:


app_df.OBS_30_CNT_SOCIAL_CIRCLE.value_counts(normalize=True)*100


# In[31]:


app_df.NAME_TYPE_SUITE.fillna(app_df.NAME_TYPE_SUITE.mode()[0],inplace=True)


# In[32]:


app_df.CNT_FAM_MEMBERS.fillna(app_df.CNT_FAM_MEMBERS.mode()[0],inplace=True)


# In[33]:


app_df.EXT_SOURCE_2.fillna(app_df.EXT_SOURCE_2.median(),inplace=True)
app_df.AMT_GOODS_PRICE.fillna(app_df.AMT_GOODS_PRICE.median(),inplace=True)
app_df.AMT_ANNUITY.fillna(app_df.AMT_ANNUITY.median(),inplace=True)
app_df.DEF_60_CNT_SOCIAL_CIRCLE.fillna(app_df.DEF_60_CNT_SOCIAL_CIRCLE.median(),inplace=True)
app_df.DEF_30_CNT_SOCIAL_CIRCLE.fillna(app_df.DEF_30_CNT_SOCIAL_CIRCLE.median(),inplace=True)
app_df.OBS_30_CNT_SOCIAL_CIRCLE.fillna(app_df.OBS_30_CNT_SOCIAL_CIRCLE.median(),inplace=True)
app_df.OBS_60_CNT_SOCIAL_CIRCLE.fillna(app_df.OBS_60_CNT_SOCIAL_CIRCLE.median(),inplace=True)
app_df.DAYS_LAST_PHONE_CHANGE.fillna(app_df.DAYS_LAST_PHONE_CHANGE.median(),inplace=True)


# In[34]:


null_cols = list(app_df.columns[app_df.isna().any()])
len(null_cols)


# In[35]:


app_df.isnull().mean()*100


# converting negative values to positive in days variable so that median isn't affected

# CONCLUSION:
# 1. NAME CONTRACT TYPE- The Applicants are receiving more of Cash Toons than Raviving bans both for Target 0 and 1
# 
# 2. CODE GENDER-Number of Female applicants are twice than that of male applicants both for Target 0 and 1
# 
# 3. FLAG OWN CAR- Most(70%) of the applicants do not own a car both for Target 0 and 1 4. FLAG OWN REALTY-Most(70%) of the applicants do not owrt a house both for Target 0 and 1
# 
# 5. NAME TYPE SUITE - Moss81%) of the applicants are Unaccompanied both for Target 0 and 1 
# 6. NAME INCOME TYPE For both Target 0 and 1. Most 51%) of the applicants are eating their income from Work
# 7. NAME EDUCATION TYPE For both Target 0 and 1, atmost 71% of the applicants have completed Secondary/secondary special education
# 
# 8. NAME FAMILY STATUS 63% of the applicants are martted for both Target 0 and 1
# 
# 9. NAME HOUSING TYPE 88% of the housing type of applicants are House apartment for both Target 0 and 1
# 
# 10. OCCUPATION TYPE-Most:31% of the applicants have other Occupation type, ace non debutters and Laborere Sales staff Drivers and core staff are not able to repay the than on time
# 
# 11. WEEKDAY APPR PROCESS START- Most of the applicant have applied the loan on Tuseday and the least on Sunday 12. ORGANIZATION TYPE Most of the Applicants are working in Business Entry Type 3. Self Employed and other Organization type

# In[36]:


app_df.DAYS_BIRTH = app_df.DAYS_BIRTH.apply(lambda x:abs(x))
app_df.DAYS_EMPLOYED = app_df.DAYS_EMPLOYED.apply(lambda x:abs(x))
app_df.DAYS_ID_PUBLISH = app_df.DAYS_ID_PUBLISH.apply(lambda x:abs(x))
app_df.DAYS_REGISTRATION = app_df.DAYS_REGISTRATION.apply(lambda x:abs(x))
app_df.DAYS_LAST_PHONE_CHANGE = app_df.DAYS_LAST_PHONE_CHANGE.apply(lambda x:abs(x))


# In[37]:


app_df["YEARS_BIRTH"] = app_df.DAYS_BIRTH.apply(lambda x: int(x//356))
app_df["YEARS_EMPLOYED"] = app_df.DAYS_EMPLOYED.apply(lambda x: int(x//356))
app_df["YEARS_REGISTRATION"] = app_df.DAYS_REGISTRATION.apply(lambda x: int(x//356))
app_df["YEARS_ID_PUBLISH"] = app_df.DAYS_ID_PUBLISH.apply(lambda x: int(x//356))
app_df["YEARS_LAST_PHONE_CHANGE"] = app_df.DAYS_LAST_PHONE_CHANGE.apply(lambda x: int(x//356))


# In[38]:


app_df.AMT_CREDIT.value_counts(normalize=True)*100


# In[39]:


app_df.AMT_CREDIT.describe()


# In[40]:


app_df["AMT_CREDIT_Category"]=pd.cut(app_df.AMT_CREDIT,[0,200000,400000,600000,800000,1000000],
                                    labels = ["very low credit","Low credit","Medium Credit","High Credit","Veru High Credit"])


# In[41]:


app_df.AMT_CREDIT_Category.value_counts(normalize=True)*100


# In[42]:


app_df["AMT_CREDIT_Category"].value_counts(normalize=True).plot.bar()
plt.show()


# In[43]:


app_df["AGE_Category"]= pd.cut(app_df.YEARS_BIRTH,[0,25,45,65,85],
                              labels=["below 25","25-45","45-65","65-85"])


# In[44]:


app_df.AGE_Category.value_counts(normalize=True)*100


# In[45]:


app_df["AGE_Category"].value_counts(normalize=True).plot.pie(autopct= '$1.2f%%')
plt.show()


# In[46]:


app_df.head()


# In[47]:


app_df.tail()


# In[48]:


tar_0 = app_df[app_df.TARGET == 0]
tar_1 = app_df[app_df.TARGET == 1]


# In[49]:


app_df.TARGET.value_counts(normalize=True)*100


# UNIVARIATE ANALYSIS
# 

# In[50]:


cat_cols = list(app_df.columns[app_df.dtypes == np.object])
num_cols = list(app_df.columns[app_df.dtypes == np.int64])+list(app_df.columns[app_df.dtypes == np.float64])


# In[51]:


cat_cols


# In[52]:


num_cols


# In[53]:


for col in cat_cols:
    print(app_df[col].value_counts(normalize=True))
    plt.figure(figsize=[5,5])
    app_df[col].value_counts(normalize=True).plot.pie(labeldistance = None,autopct = '%1.2f%%')
    plt.legend()
    plt.show()


# In[54]:


num_cols_withoutflag=[]
num_cols_withflag=[]
for col in num_cols:
    if col.startswith("FLAG"):
        num_cols_withflag.append(col)
    else:
        num_cols_withoutflag.append(col)


# In[55]:


num_cols_withoutflag


# In[56]:


num_cols_withflag


# In[57]:


for col in num_cols_withoutflag:
    print(app_df[col].describe())
    plt.figure(figsize=[8,5])
    sns.boxplot(data = app_df,x=col)
    plt.show()
    print("---------------")


# In[58]:


for col in cat_cols:
    print(f"plot on {col} for target 0 and 1")
    plt.figure(figsize=[10,7])
    plt.subplot(1,2,1)
    tar_0[col].value_counts(normalize=True).plot.bar()
    plt.title("target 0")
    plt.xlabel(col)
    plt.ylabel("Density")
    tar_1[col].value_counts(normalize=True).plot.bar()
    plt.title("target 1")
    plt.xlabel(col)
    plt.ylabel("Density")
    plt.show()
    print("\n\n----------------------------------------\n\n")


# In[59]:


plt.figure(figsize=(10,6))
sns.distplot(tar_0['AMT_GOODS_PRICE'],label='tar_0',hist=False)
sns.distplot(tar_1['AMT_GOODS_PRICE'],label='tar_1',hist=False)
plt.legend()
plt.show()


# BIVARIATE AND MULTIVARIATE ANALYSIS
# 

# In[60]:


plt.figure(figsize=[15,10])
plt.subplot(1,2,1)
sns.boxplot(x="WEEKDAY_APPR_PROCESS_START",y="HOUR_APPR_PROCESS_START",data=tar_0)
plt.subplot(1,2,2)
sns.boxplot(x="WEEKDAY_APPR_PROCESS_START",y="HOUR_APPR_PROCESS_START",data=tar_1)
plt.show()


# In[61]:


plt.figure(figsize=[15,10])
plt.subplot(1,2,1)
sns.boxplot(x="AGE_Category",y="AMT_CREDIT",data=tar_0)
plt.subplot(1,2,2)
sns.boxplot(x="AGE_Category",y="AMT_CREDIT",data=tar_1)
plt.show()


# In[62]:


sns.pairplot(tar_0[["AMT_INCOME_TOTAL","AMT_ANNUITY","AMT_GOODS_PRICE"]])
plt.show()


# In[63]:


sns.pairplot(tar_1[["AMT_INCOME_TOTAL","AMT_CREDIT","AMT_ANNUITY","AMT_GOODS_PRICE"]])
plt.show()


# Co-relation between numerical columns
# 

# In[64]:


corr_data = app_df[["AMT_INCOME_TOTAL","AMT_CREDIT","AMT_ANNUITY","AMT_GOODS_PRICE","YEARS_BIRTH","YEARS_EMPLOYED","YEARS_REGISTRATION","YEARS_ID_PUBLISH","YEARS_LAST_PHONE_CHANGE"]]
corr_data.head()


# In[65]:


plt.figure(figsize=[10,10])
sns.heatmap(corr_data.corr(),annot=True,cmap="RdYlGn")
plt.show()


# CONCLUSION:
# 1. ANT INCOME TOTAL - It is less correlated with AMT CREDIT AMT ANNUITY AMT GOODS_PRICE respectively
# 
# 2. ANT CREDIT- Is has a strong positive coreltaion Index of 0.56.0.75 with AMT GOODS PRICE, AMT ANNUITY respectively and also positive covetation with other Year Columns 
# 3. ANT ANNUITY - Is has positive coreitaion index of 0.75 with AMT_CREDIT, AMT GOODS PRICE and Negative with YEAR EMPLOYED YEAR REGISTRATION
# 
# 4. AMT GOODS PRICE-It has a strong positive corelation index 0.75.0.58 with AMT ANNUITY AMT CREDIT and weak positive corelation with otherYear columns

# spliting the numerical variables based on target 0 and target 1 to find corelation

# In[66]:


corr_data_0 = tar_0[["AMT_INCOME_TOTAL","AMT_CREDIT","AMT_ANNUITY","AMT_GOODS_PRICE","YEARS_BIRTH","YEARS_EMPLOYED","YEARS_REGISTRATION","YEARS_ID_PUBLISH","YEARS_LAST_PHONE_CHANGE"]]
corr_data_0.head()


# In[67]:


corr_data_1 = tar_1[["AMT_INCOME_TOTAL","AMT_CREDIT","AMT_ANNUITY","AMT_GOODS_PRICE","YEARS_BIRTH","YEARS_EMPLOYED","YEARS_REGISTRATION","YEARS_ID_PUBLISH","YEARS_LAST_PHONE_CHANGE"]]
corr_data_1.head()


# In[68]:


plt.figure(figsize=[10,10])
sns.heatmap(corr_data_0.corr(),annot=True,cmap="RdYlGn")
plt.show()


# In[69]:


plt.figure(figsize=[10,10])
sns.heatmap(corr_data_1.corr(),annot=True,cmap="RdYlGn")
plt.show()


# ##READ PREVIOUS APPLICATION CSV

# In[70]:


papp_data=pd.read_csv("previous_application.csv")
papp_data


# In[71]:


papp_data.info()


# In[72]:


papp_data.describe()


# In[73]:


papp_data.isnull().mean()*100


# In[74]:


percentage = 49
threshold_p = int(((100-percentage)/100)*papp_data.shape[0]+1)
papp_df=papp_data.dropna(axis=1, how='any')
papp_df=papp_data.dropna(axis=1, thresh=threshold_p)
papp_df.head()


# In[75]:


papp_data.shape


# In[76]:


for col in papp_df.columns:
    if papp_df[col].dtypes == np.int64 or papp_df[col].dtypes == np.float64:
        papp_df[col]=papp_df[col].apply(lambda x:abs(x))


# In[77]:


null_cols = list(papp_df.columns[papp_df.isna().any()])
len(null_cols)


# In[78]:


papp_df.isnull().mean()*100


# In[79]:


papp_df.AMT_CREDIT.describe()


# In[80]:


papp_df["AMT_CREDIT_Category"]=pd.cut(papp_df.AMT_CREDIT,[0,200000,400000,600000,800000,1000000],
                                    labels = ["very low credit","Low credit","Medium Credit","High Credit","Veru High Credit"])


# In[81]:


papp_df["AMT_CREDIT_Category"].value_counts(normalize=True).plot.bar()
plt.show()


# In[82]:


approved = papp_df[papp_df.NAME_CONTRACT_STATUS == "APPROVED"]
cancelled = papp_df[papp_df.NAME_CONTRACT_STATUS == "cancelled"]
unused = papp_df[papp_df.NAME_CONTRACT_STATUS == "Unused"]
refused = papp_df[papp_df.NAME_CONTRACT_STATUS == "REFUSED"]


# In[83]:


papp_df.NAME_CONTRACT_STATUS.value_counts(normalize=True)*100


# In[84]:


papp_df.NAME_CONTRACT_STATUS.value_counts(normalize=True).plot.pie(autopct='%1.2f%%')
plt.legend()
plt.show()


# In[85]:


cat_cols = list(papp_df.columns[papp_df.dtypes == np.object])
num_cols = list(papp_df.columns[papp_df.dtypes == np.int64])+list(papp_df.columns[papp_df.dtypes == np.float64])


# In[86]:


cat_cols


# In[87]:


num_cols


# In[88]:


cat_cols=["NAME_CONTRACT_TYPE","WEEKDAY_APPR_PROCESS_START","NAME_CONTRACT_STATUS","NAME_PAYMENT_TYPE","NAME_CLIENT_TYPE","NAME_SELLER_INDUSTRY","CHENNEL_TYPE","NAME_YIELD_GROUP","PRODUCT_COMBINATION"]


# In[99]:


num_cols=["HOUR_APPR_PROCESS_START","AMT_ANNUITY","AMT_APPLICATION","AMT_CREDIT","AMT_GOODS_PRICE","CNT_PAYMENT","DAYS_DECISION"]


# In[102]:


for col in cat_cols:
    print(papp_df[col].value_counts(normalize=True)*100)
    plt.figure(figsize=[5,5])
    papp_df[col].value_counts(normalize=True).plot.pie(labeldistance=None,autopct = '%1.2f%%')
    plt.legend()
    plt.show()
    print("---------------")


# In[93]:


for col in num_cols:
    print("99th Percentile",np.percentile(papp_df[col],99))
    print(papp_df[col].describe())
    plt.figure(figsize=[10,6])
    sns.boxplot(data=papp_df,x=col)
    plt.show()
    print("-------------------")


# Bivariate and multivariate analysis
# 

# In[98]:


plt.figure(figsize=[15,10])
plt.subplot(1,4,1)
plt.title("APPROVED")
sns.scatterplot(x='AMT_ANNUITY',y='AMT_GOODS_PRICE',data=approved)
plt.subplot(1,4,2)
plt.title("CANCELLED")
sns.scatterplot(x='AMT_ANNUITY',y='AMT_GOODS_PRICE',data=cancelled)
plt.subplot(1,4,3)
plt.title("REFUSED")
sns.scatterplot(x='AMT_ANNUITY',y='AMT_GOODS_PRICE',data=refused)
plt.subplot(1,4,4)
plt.title("UNUSED")
sns.scatterplot(x='AMT_ANNUITY',y='AMT_GOODS_PRICE',data=unused)
plt.show()


# CONCLUISON:
# 1. for loan status as Approved,Refused,Unused,Cancelled AMount  of annuilty increases with goods price
# 2. foa losn status as Refused it has no linear relationship

# In[103]:


corr_approved = approved[["DAYS_DECISION","AMT_ANNUITY","AMT_APPLICATION","AMT_CREDIT","AMT_GOODS_PRICE","CNT_PAYMENT"]]
corr_cancelled = cancelled[["DAYS_DECISION","AMT_ANNUITY","AMT_APPLICATION","AMT_CREDIT","AMT_GOODS_PRICE","CNT_PAYMENT"]]
corr_refused = refused[["DAYS_DECISION","AMT_ANNUITY","AMT_APPLICATION","AMT_CREDIT","AMT_GOODS_PRICE","CNT_PAYMENT"]]
corr_unused = unused[["DAYS_DECISION","AMT_ANNUITY","AMT_APPLICATION","AMT_CREDIT","AMT_GOODS_PRICE","CNT_PAYMENT"]]


# In[104]:


plt.figure(figsize=[10,10])
sns.heatmap(corr_approved.corr(),annot=True,cmap="Blues")
plt.title("Heat Map plot for Approved")
plt.show()


# In[105]:


plt.figure(figsize=[10,10])
sns.heatmap(corr_unused.corr(),annot=True,cmap="Blues")
plt.title("Heat Map plot for unused")
plt.show()


# In[106]:


plt.figure(figsize=[10,10])
sns.heatmap(corr_refused.corr(),annot=True,cmap="Blues")
plt.title("Heat Map plot for Refused")
plt.show()


# merger the application nad previous application data frames
# 

# In[107]:


merge_df = app_df.merge(papp_df,on=["SK_ID_CURR"],how = 'left')
merge_df.head()


# In[108]:


merge_df.info()


# FILTERING REQUIRED COLUMNS

# In[109]:


for col in merge_df.columns:
    if col.startswith("FLAG"):
        merge_df.drop(columns=col,axis=1,inplace=True)


# In[110]:


merge_df.shape


# In[111]:


res1 = pd.pivot_table(data=merge_df,index=["NAME_INCOME_TYPE","NAME_CLIENT_TYPE"],columns=["NAME_CONTRACT_STATUS"],values="TARGET",aggfunc="mean")


# In[112]:


res1


# In[113]:


plt.figure(figsize=[10,10])
sns.heatmap(res1,annot=True,cmap='BuPu')
plt.show()


# The final Conclusion
# 
# *Applicants with income type maternity leave and client type New are having more chances of getting the loan approved .\
# 
# *Applicants with inocme type Matenrity Leave,Unemployed and client type Repeater are having getting the loan cancelled .
# 
# *Applicants with income type maternity leave and client type New are having more chances of getting the loan Refused .
# 
# *Applicants with inocme type Matenrity Leave,and client type Repeater,working and client type new are not able to utilize the banks offer.

# In[114]:


res2 = pd.pivot_table(data = merge_df,index=["CODE_GENDER","NAME_SELLER_INDUSTRY"],columns=["TARGET"],values="AMT_GOODS_PRICE_x",aggfunc="sum")


# In[115]:


res2


# In[116]:


plt.figure(figsize=[13,13])
sns.heatmap(res2,annot=True,cmap='BuPu')
plt.show()


# THUS WE CONCLUDE OUR PROJECT -CREDIT EDA
