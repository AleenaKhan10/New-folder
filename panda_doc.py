import numpy as np
import pandas as pd

#______________Object Creation______________

# Creating the series(1D Arrays)
s = pd.Series([1,2,3,4,np.nan,5,6])
# print(s)


# Creating the Dataframes
dates = pd.date_range(start='20230601', periods=9)
# print(dates)
df = pd.DataFrame(data = np.random.randn(9, 5), index=dates, columns=list("ABCDE"))
# print(df)


# Creating DataFrame using Dictionary
df2 = pd.DataFrame({
    'A': pd.Timestamp('20230601'),
    'B': 1.0,
    'C': np.array([3]*4),
    'D': ['test', 'drill', 'drill', 'test']    
})
# print(df2.dtypes)


# ______________Viewing Data______________

'''we can view the data by using the following methods:
    1- head()        => Respresents the uppers values
    2- tail()        => Respresents the lower values
    3- index()       => Respresents the index
    4- column()      => Respresents the headings
    5- to_numpy()    => Respresents the underlying data
    5- Describe()    => Respresents the Statistical summary of the data
we can also sort them using the sort_index() or sort_values() methods'''
    
df.head()
df.tail()
df.index
df.columns
df.to_numpy()
df.describe()
df.sort_index(axis=0, ascending=False)
df.sort_values("B")

# ______________Selection______________

'''we can select the data by using the following methods:
    1- Selecting via [] (__getitem__)
    2- Selection by label (.loc(), .at())
    3- Selection by position (.iloc(), .iat())
    4- Boolean indexing'''
    
(df.loc[:'20230604' , ['A','B']])  #for fast access to a scalar we can use df.at()
(df.iloc[:4 , 1:3])  #for fast access to a scalar we can use df.iat()
(df[df['A'] < 0])

# we can also use the isin() method for filtering
df2 = df.copy()
df2['F'] = ['one', 'one', 'two', 'two', 'one', 'three', 'one', 'two', 'three']
(df2[df2['F'].isin(['one', 'three'])])

# we can also change the values by selecting the data with above methods
df.iloc[:1, :1] = 0
df.loc[:'20230601', ['B']] = 0
(df)

# ______________Handling missing data______________
'''we can hadle the missing data by the following methods:
    1- dropna()   => drops all the rows with missing values
    2- fillna()   => fill all the missing values
    3- isna()     => gets true where data is nan '''
 
# first adding new column to the Dataframe object
df3 = df.reindex(index=dates[:4], columns=list(df.columns) + ['F'])
df3.loc[dates[0]:dates[1] , 'F'] = 1
(df3)

(df3.dropna())
(df3.fillna(value = 5))
(df3.isna())


# ______________Operations______________
'''we can also apply some functions like .sub() that nan all the values in a row if there is one nan value or we can 
find the mean() by give the axis. We can also use the apply() method to apply some user build funtions.'''


# ______________Merge______________
'''we can merge two series or dataframes by using concat() and merge() methods'''

left = pd.DataFrame({"key":["foo","bar"], "lvalue":[1,2]})
right = pd.DataFrame({"key":["foo","bar"], "rvalue":[3,4]})

merged = pd.merge(left, right, on="key")
print(merged)