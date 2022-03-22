from cProfile import label
from pprint import pprint
from matplotlib.pyplot import axis
import numpy as np
import pandas as pd
from numpy.random import randn as rn

l = [1,2,3,4,5]
label = ['a','b','c','d','e']
ar = np.array(l)
# print(ar, type(ar))
d = {'a':1,'b':2,'c':3}
# print(d)

# s = pd.Series(data=l)
# print(type(s))

# s = pd.Series(data=l,index=label)
# s = pd.Series(l,label)

# type of values can a Pandas Series 
# s1 = pd.Series(ar)
# s2 = pd.Series(l)
# s3 = pd.Series(label)
# s4 = pd.Series(d)
# s5 = pd.Series(data=[sum,print,max])
# s6 = pd.Series(data=[d.keys,d.values,d.items])

# print(s1)
# print(s2)
# print(s3)
# print(s4)
# print(s5)
# print(s6)

# Indexing and slicing
# ser1 = pd.Series([1,2,3,4],['CA', 'OR', 'CO', 'AZ'])
# ser2 = pd.Series([1,2,5,4],['CA', 'OR', 'NV', 'AZ'])
# print(ser1)
# print(ser2)
# print(ser1['CA'])
# print(ser2['NV'])
# print(len(ser2))
# print(ser2[:len(ser2)-1])
# print(ser1[1:3])

# Adding/Merging two series with common indices
# ser1 = pd.Series([1,2,3,4],['CA', 'OR', 'CO', 'AZ'])
# ser2 = pd.Series([1,2,5,4],['CA', 'OR', 'NV', 'AZ'])
# ser3 = ser1+ser2
# ser4 = ser1*ser2
# print(ser3)

np.random.seed(100)
matt = rn(5,4)
r_label = ['a','b','c','d','e']
col = ['v','x','y','z']

df = pd.DataFrame(data=matt,index=r_label, columns=col)
# print(df)
# print(df['x'])
# print(df[['v','y']])
# print(df.x)

# Creating and deleting a (new) column or row
df['new'] = df['v']+df['z']
# df = df.drop('new',axis=1)
# df = df.drop('e')
# print(df)

# Selecting/indexing Rows
# print(df.loc['d'])
# print(df.loc[['b','c']])
# print(df.iloc[2])
# print(df.iloc[[1,3]])

# Subsetting
# print(df.loc[['b','d'],['x','y']])

# Conditional selection, index (re)setting, multi-index
# print(df>0)
# print(df.loc[['a','b','c','d']]>0)
# booldf = df>0
# print(df[booldf])


# Passing Boolean series to conditionally subset the DataFrame
# matrix_data = np.matrix('22,66,140;42,70,148;30,62,125;35,68,160;25,62,152')
# row_labels = ['A','B','C','D','E']
# column_headings = ['Age', 'Height', 'Weight']
# df1 = pd.DataFrame(data=matrix_data, index = row_labels,columns=column_headings)
# print(df1[df1['Height']>65])
# print(df1[df1['Age'] > 25])
# print(df1[(df1['Age'] > 25) & (df1['Age'] < 40)])
# print(df1[df1['Age'].between(25, 38)].sort_values(by=['Age']))

# Resetting and setting index
# print(df1)
# print(df1.reset_index())
# print(df1.reset_index(drop=True))
# print(df1.set_index('Age'))

# Multi-indexing
# outside = ['G1','G1','G1','G2','G2','G2']
# inside = [1,2,3,1,2,3]
# hier_index = list(zip(outside,inside))

# print("\nTuple pairs after the zip and list command\n",'-'*45, sep='')
# print(hier_index)
# hier_index = pd.MultiIndex.from_tuples(hier_index)
# print("\nIndex hierarchy\n",'-'*25, sep='')
# print(hier_index)
# print("\nIndex hierarchy type\n",'-'*25, sep='')
# print(type(hier_index))

# print("\nCreating DataFrame with multi-index\n",'-'*37, sep='')
# np.random.seed(101)
# df1 = pd.DataFrame(data=np.round(rn(6,3),2), index= hier_index, columns= ['A','B','C'])
# print(df1)

# print("\nSubsetting multi-index DataFrame using two 'loc' methods\n",'-'*60, sep='')
# print(df1.loc['G2'])

# print("\nNaming the indices by 'index.names' method\n",'-'*45, sep='')
# df1.index.names=['Outer', 'Inner']
# print(df1)

# Cross-section ('XS') command
# print(df1.xs('G1'))
# print(df1.xs(2,level='Inner'))

# Missing Values
# df = pd.DataFrame({'A':[1,2,np.nan],'B':[5,np.nan,np.nan],'C':[1,2,3]})
# df['States']="CA NV AZ".split()
# df.set_index('States',inplace=True)
# print(df)

# Pandas 'dropna' method
# print(df.dropna(axis=0))
# print(df.dropna(axis=1))
# print(df.dropna(axis=0, thresh=2))

# Pandas 'fillna' method
# print(df.fillna(value='Filling'))
# print(df.fillna(value=df['A'].mean()))

# GroupBy method
# data = {'Company':['GOOG','GOOG','MSFT','MSFT','FB','FB'],
#        'Person':['Sam','Charlie','Amy','Vanessa','Carl','Sarah'],
#        'Sales':[200,120,340,124,243,350]}
# df = pd.DataFrame(data)
# comp = df.groupby('Company')
# print(comp.mean())
# print(comp.sum())
# print(pd.DataFrame(df.groupby('Company').describe().loc['FB']).transpose())
# print(df.groupby('Company').describe().loc[['GOOG', 'MSFT']])

# Merging, Joining, Concatenating
# df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
#                         'B': ['B0', 'B1', 'B2', 'B3'],
#                         'C': ['C0', 'C1', 'C2', 'C3'],
#                         'D': ['D0', 'D1', 'D2', 'D3']},
#                         index=[0, 1, 2, 3])

# df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
#                         'B': ['B4', 'B5', 'B6', 'B7'],
#                         'C': ['C4', 'C5', 'C6', 'C7'],
#                         'D': ['D4', 'D5', 'D6', 'D7']},
#                          index=[4, 5, 6, 7])

# df3 = pd.DataFrame({'A': ['A8', 'A9', 'A10', 'A11'],
#                         'B': ['B8', 'B9', 'B10', 'B11'],
#                         'C': ['C8', 'C9', 'C10', 'C11'],
#                         'D': ['D8', 'D9', 'D10', 'D11']},
#                         index=[8,9,10,11])

# print(df1)
# print(df2)
# print(df3)
# df_cat = pd.concat([df1,df2,df3],axis=0)
# df_cat1 = pd.concat([df1,df2,df3],axis=1)
# df_cat1.fillna(value=0,inplace=True)
# print(df_cat1)

# Merging by a common 'key'
# left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
#                      'A': ['A0', 'A1', 'A2', 'A3'],
#                      'B': ['B0', 'B1', 'B2', 'B3']})
   
# right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
#                           'C': ['C0', 'C1', 'C2', 'C3'],
#                           'D': ['D0', 'D1', 'D2', 'D3']})

# # print(left)
# # print(right)

# df_merge = pd.merge(left,right,how='left',on='key')
# print(df_merge)

# left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
#                      'key2': ['K0', 'K1', 'K0', 'K1'],
#                         'A': ['A0', 'A1', 'A2', 'A3'],
#                         'B': ['B0', 'B1', 'B2', 'B3']})
    
# right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
#                                'key2': ['K0', 'K0', 'K0', 'K0'],
#                                   'C': ['C0', 'C1', 'C2', 'C3'],
#                                   'D': ['D0', 'D1', 'D2', 'D3']})
# df_merge= pd.merge(left, right, on=['key1', 'key2'])
# df_merge1=pd.merge(left, right, how='outer',on=['key1', 'key2'])
# df_left=pd.merge(left, right, how='left',on=['key1', 'key2'])
# df_right = pd.merge(left,right,how='right',on=['key1','key2'])
# print(df_right)

# left = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
#                      'B': ['B0', 'B1', 'B2']},
#                       index=['K0', 'K1', 'K2']) 

# right = pd.DataFrame({'C': ['C0', 'C2', 'C3'],
#                     'D': ['D0', 'D2', 'D3']},
#                       index=['K0', 'K2', 'K3'])

# print(left.join(right))

# Useful operations
# df = pd.DataFrame({'col1':[1,2,3,4,5,6,7,8,9,10],
#                    'col2':[444,555,666,444,333,222,666,777,666,555],
#                    'col3':'aaa bb c dd eeee fff gg h iii j'.split()})
# print(df.head())
# print(df['col2'].unique())
# print(df['col2'].nunique())
# print(df['col2'].value_counts())

# apply function
# def testfunc(x):
#     if (x> 500):
#         return (10*np.log10(x))
#     else:
#         return (x/10)

# df['new'] = df['col2'].apply(testfunc)
# df['col_len'] = df['col3'].apply(len)
# df['sqrt'] = df['new'].apply(lambda x:np.sqrt(x))

# statistical function
# print(df['new'].sum())
# print(df['new'].max())
# print(df['new'].min())
# print(df['new'].std())

# Get column names and sort values
# print(list(df.columns))
# del df['col_len']
# print(df.sort_values(by='col2'))
# print(df.sort_values(by='col2',ascending=False))

# df = pd.DataFrame({'col1':[1,2,3,np.nan],
#                    'col2':[np.nan,555,666,444],
#                    'col3':['abc','def','ghi','xyz']})
# print(df.head())
# print(df.isnull())
# print(df.fillna('FILL'))

# Pivot table
# data = {'A':['foo','foo','foo','bar','bar','bar'],
#      'B':['one','one','two','two','one','one'],
#        'C':['x','y','x','y','x','y'],
#        'D':[1,3,2,5,4,1]}

# df = pd.DataFrame(data)
# print(df)
# print(df.pivot_table(values='D',index=['A', 'B'],columns=['C']))
# print(df.pivot_table(values='D',index=['A', 'B'],columns=['C'], fill_value='FILLED'))

mylist=[1,2,3,4,5,6,7,8,9,10]
print(np.percentile(mylist,25))
print(np.percentile(mylist,90))