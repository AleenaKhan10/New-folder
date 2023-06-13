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