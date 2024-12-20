import pandas as pd
import random
#import numpy as np
from sklearn.preprocessing import MinMaxScaler

dfs = pd.DataFrame({})
#For the week from 4Nov-10Nov:
for i in range(4,5):
    print("Reading... "+'./trento/sms-call-internet-tn-2013-11-0{}.csv'.format(i))
    df = pd.read_csv('./trento/sms-call-internet-tn-2013-11-0{}.csv'.format(i), parse_dates=['time'])
    dfs = pd.concat([dfs,df])
#df = pd.read_csv('./trento/sms-call-internet-tn-2013-11-10.csv', parse_dates=['time'])
#dfs = pd.concat([dfs,df])
dfs = dfs.fillna(0)
del df

#For the whole data available: (modify as per research needs)
#for i in range(1,10):
#    print("Reading... "+'./milan/sms-call-internet-mi-2013-11-0{}.csv'.format(i))
#    df = pd.read_csv('./milan/sms-call-internet-mi-2013-11-0{}.csv'.format(i), parse_dates=['time'])
#    dfs = pd.concat([dfs,df])
#for i in range(10,31):
#    df = pd.read_csv('./milan/sms-call-internet-mi-2013-11-{}.csv'.format(i), parse_dates=['time'])
#    dfs = pd.concat([dfs,df])
#for i in range(1,10):
#    df = pd.read_csv('./milan/sms-call-internet-mi-2013-12-0{}.csv'.format(i), parse_dates=['time'])
#    dfs = pd.concat([dfs,df])
#for i in range(10,32):
#    df = pd.read_csv('./milan/sms-call-internet-mi-2013-12-{}.csv'.format(i), parse_dates=['time'])
#    dfs = pd.concat([dfs,df])
#df = pd.read_csv('./milan/sms-call-internet-mi-2014-01-01.csv', parse_dates=['time'])
#dfs = pd.concat([dfs,df])
#dfs = dfs.fillna(0)
#del df
print("Grouping by hours...") 
#Group by hours and aggregate values according to each cell/hour
dfgr = dfs[['cellid', 'time', 'smsin','smsout', 'callin','callout', 'internet']].groupby(['time', 'cellid'], as_index=False).sum()
dfgr['hour'] = dfgr.time.dt.hour+24*(dfgr.time.dt.day-min(dfs.time.dt.day))+((30*24)*(dfgr.time.dt.month-11))+(361*24*(dfgr.time.dt.year-2013))
dfgg = dfgr[['hour', 'cellid', 'smsin','smsout', 'callin','callout', 'internet']].groupby(['hour', 'cellid'], as_index=False).sum()
#dfgg = dfgg.set_index(['hour']).sort_index()
del dfs

##Random grid (new-trento)
unique_values = dfgg['cellid'].unique()
unique_df = pd.DataFrame(unique_values, columns=['unique_values'])
random_unique_values = unique_df.sample(n=min(250, len(unique_df)), random_state=123)

rl = random_unique_values['unique_values'].tolist()

##10x10 grid
#l=[] #choosen cells on 10x10 grid
#for i in range(4379,5549,117): #4022,6245
#    for j in range(i,i+10):
#        l.append(j)
#print(l)

##12x12 grid
#l=[] #choosen cells on 12x12 grid
#for i in range(4379,5783,117): #4022,6245
#    for j in range(i,i+12):
#        l.append(j)
#print(l)

##15x15 grid
l=[] #choosen cells on 15x15 grid
for i in range(4379,6134,117): #4022,6245
    for j in range(i,i+15):
        l.append(j)
print(l)

for i in range(5997, 6465, 117):
     for j in range(i,i+4):
          l.append(j)
#l is a 18x18 grid

#Start creating the 12x12 dataframe
df20 = pd.DataFrame({})
for el in dfgg.iterrows():
    if(l.count(int(el[1].cellid))==1):
        tmp = pd.DataFrame([[int(el[1].hour), int(el[1].cellid), el[1].smsin, el[1].smsout, el[1].callin, el[1].callout, el[1].internet]], columns=['hour', 'cellid', 'smsin','smsout', 'callin','callout', 'internet'])
        df20=pd.concat([df20,tmp])
df20 = df20.set_index(['hour']).sort_index()
cells = df20.cellid
#dfgg = dfgg.set_index(['hour']).sort_index()
#cells = dfgg.cellid
scaler=MinMaxScaler()
#scaled = pd.DataFrame(scaler.fit_transform(dfgg), columns=dfgg.columns, index=dfgg.index)
scaled = pd.DataFrame(scaler.fit_transform(df20), columns=df20.columns, index=df20.index)
dfgg = scaled
dfgg['cellid']=cells
#dfgg.to_csv('./classif/tn_fg_temp_day.csv', index="hour")
print("Temporary save completed.")
#temp save of the current data

# Ahora usamos pivot_table para transformar los datos
#df_pivot = dfgg.pivot_table(index='cellid', columns='hour', values=['smsin', 'smsout', 'callin', 'callout', 'internet'], aggfunc='sum')

# Aplanamos las columnas multi-nivel
#df_pivot.columns = [f'{col[0]}_{col[1]}' for col in df_pivot.columns]

# Reiniciamos el índice para que 'cellid' sea una columna
#df_pivot.reset_index(inplace=True)
#df_pivot = df_pivot.fillna(0)
#df_pivot.to_csv("./csv/tn/classif_10x10.csv")

#exit()

##Random grid (milan)
#l=[]
#random.seed(13479561) #To make it reproducible
#for i in range(5000,10000,10):
    #print (random.randint(i,i+100))
#    a_string = str(i)
#    a_length = len(a_string)
#    c = int(a_string[a_length - 2: a_length])
#    if(c<50):
#        continue
#    else:
#        l.append(random.randint(i,i+10))

#complete the dataframe
dfsi = [] #data from df without indexing
dfsi2 = pd.DataFrame({}) #new df to save
dfso = []
dfso2 = pd.DataFrame({})
dfci = []
dfci2 = pd.DataFrame({})
dfco = []
dfco2 = pd.DataFrame({})
dfin = []
dfin2 = pd.DataFrame({})

dfsi.append(dfgg.loc[0].cellid)
dfso.append(dfgg.loc[0].cellid)
dfci.append(dfgg.loc[0].cellid)
dfco.append(dfgg.loc[0].cellid)
dfin.append(dfgg.loc[0].cellid)

print(dfgg)

dfsi2["cellid"]=dfsi[0].values
dfso2["cellid"]=dfsi[0].values
dfci2["cellid"]=dfsi[0].values
dfco2["cellid"]=dfsi[0].values
dfin2["cellid"]=dfsi[0].values

print(dfgr.hour.max())
for i in range(0, dfgr.hour.max()+1): #dfgr.hour.max()+1
    
        #row[0] - hour, row[1]: df cols
        dfsi.append(dfgg.loc[i].smsin)
        dfsi2["smsin"+str(i)]=dfsi[i+1].values
        
        dfso.append(dfgg.loc[i].smsout)
        dfso2["smsout"+str(i)]=dfso[i+1].values
        
        dfci.append(dfgg.loc[i].callin)
        dfci2["callin"+str(i)]=dfci[i+1].values
        
        dfco.append(dfgg.loc[i].callout)
        dfco2["callout"+str(i)]=dfco[i+1].values
        
        dfin.append(dfgg.loc[i].internet)
        dfin2["internet"+str(i)]=dfin[i+1].values
    #except: #if any hour have some cells without data, go to the next hour
    #    continue

print("Saving results...")
dfsi2.to_csv("./csv/19x19/classif_18x18_smsin.csv")
dfso2.to_csv("./csv/19x19/classif_18x18_smsout.csv")
dfci2.to_csv("./csv/19x19/classif_18x18_callin.csv")
dfco2.to_csv("./csv/19x19/classif_18x18_callout.csv")
dfin2.to_csv("./csv/19x19/classif_18x18_internet.csv")
