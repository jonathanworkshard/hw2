#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 15:07:42 2018

@author: jsshenkman
"""



import numpy as np
import pandas as pd
import psycopg2
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import sys

old_stdout = sys.stdout

log_file = open("hw2_3.log","w")

sys.stdout = log_file


"""
THIS IS HOW I QUERIED THE DATA.  FOR REPLICATION, I STORED THE RESULTS LOCALLY AND LOADED EACH TIME I RAN

conn = psycopg2.connect("dbname='iems308' user='jss027' host='gallery.iems.northwestern.edu' password='jss027_pw'")
data_pract = pd.read_sql("SELECT * FROM pos.trnsact WHERE c6 = '2005-06-03';", con=conn)
data_pract.to_pickle('/Users/jsshenkman/Documents/python/data_pract1.pkl')
"""

#cur = conn.cursor()


#data = pd.read_sql('SELECT * FROM pos.strinfo',conn)

#trans_data_ex = pd.read_sql('SELECT * FROM pos.trnsact LIMIT 10', con=conn)

table_names = list(['deptinfo', 'skstinfo', 'skuinfo', 'strinfo', 'trnsact'])

base_data = {}
trnsact_columns= {'c1':'sku', 
                       'c2':'store', 
                       'c3':'register',
                       'c4':'trannum', 
                       'c5':'seq', 
                       'c6':'saledate', 
                       'c7':'stype',
                       'c8':'quantity',
                       'c9':'orgprice', 
                       'c10':'useless',
                       'c11':'sprice',
                       'c12':'interid',
                       'c13':'mic',
                       'c14':'junk'}




#dist_baskets = [pd.read_sql('SELECT * FROM pos.trnsact limit 1', con=conn, coerce_float=True)for i in np.arange(500)]
#data_pract = pd.concat(dist_baskets)


data_pract = pd.read_pickle('/Users/jsshenkman/Documents/python/data_pract1.pkl')




# prove that c5 is seq and not c12
print('number of data points is',data_pract.shape[0])
# show number of points with unique c12
print('number of unique c12', data_pract.drop_duplicates('c12').shape[0])
# get unique c5
print('number of unique c5',  data_pract.drop_duplicates('c5').shape[0])


# filter out the returns
not_return = [element.startswith('P') for element in data_pract['c7']]
data_pract = data_pract[not_return]



# convert sprice to number
data_pract['c11'] = pd.to_numeric(data_pract['c11'])


# look at the busiest stores
store_revenue = data_pract.groupby('c2')['c11'].sum().sort_values(ascending=False)
print('total number of stores is', len(store_revenue))

# take just the top stores
num_stores = 40
top_stores = list(store_revenue[:num_stores].keys())

# get the portion of total revenue coming from these stores
perc_top = np.sum(store_revenue[:num_stores])/np.sum(store_revenue)
print('portion of revenue coming from top stores', )


# convert quantity to number
data_pract['c8'] = pd.to_numeric(data_pract['c8'])


"""
# get just the unique baskets
unique_baskets = data_pract.drop_duplicates(['c2', 'c3', 'c4','c5'])[['c2', 'c3', 'c4','c5']]
#unique_baskets.index = np.arange(len(unique_baskets))

# select random baskets
num_baskets = 20
basket_index =np.random.randint(0,unique_baskets.shape[0],num_baskets)



# group data to see number of sku bought
basket_table = data_pract.groupby(['c2','c3','c4','c5','c1'])['c8'].sum()

# adjust index to only see the baskets (drop sku)
new_index = basket_table.index.droplevel(-1)

# get indexes where each basket appears in the grouping
group_index_choose = [np.where((list(new_index) == unique_baskets.iloc[i].values).all(axis=1)) for i in basket_index]
"""

# set random 
# basket_table = data_pract[['c1','c2', 'c3', 'c4','c5','c8']].to_sparse().groupby(['c2','c3','c4','c5','c1'],sort=False,squeeze=True)['c8'].sum().unstack()
# basket_table = data_pract.groupby(['c2','c3','c4','c5','c1'])['c8'].sum()[unique_baskets.iloc[basket_index].apply(tuple,axis=1)].unstack().fillna(0)

# get the onehotencoder format for transactions that occured at the top 10 revenue stores
basket_table = data_pract.groupby(['c2','c3','c4','c5','c1'])['c8'].sum()[top_stores].unstack().fillna(0)


# convert to binary
def make_binary(x):
    if x >= 1:
        return 1
    else:
        return 0

basket_table_binary = basket_table.applymap(make_binary)

# run api
min_sup = apriori(basket_table_binary, min_support=0.0001, use_colnames=True)
evaluated = association_rules(min_sup, metric="lift", min_threshold=1)
recommended_moves = evaluated.sort_values('support',ascending=False)[evaluated['confidence']>=.6]

def strip_whitespace(x):
    return x.strip()

def out_frozen_set(x):
    return list(x)[0]


recommended_moves.iloc[:,[0,1]] = recommended_moves.iloc[:,[0,1]].applymap(out_frozen_set).applymap(strip_whitespace)

num_implications = recommended_moves.groupby('antecedants')['lift'].count().sort_values(ascending=False)




sku_bunch = recommended_moves.iloc[:,[0,1]]
print('number of recommended items to move is', len(np.unique(sku_bunch)))


"""
sku_dict = {}
for index,basket in unique_baskets.iterrows():
    sku_index = np.where((data_pract[['c2', 'c3', 'c4','c5']] == basket).all(axis=1))[0]
    sku_dict[index] = data_pract['c1'].iloc[sku_index].values
"""    

# finish logging
sys.stdout = old_stdout

log_file.close()