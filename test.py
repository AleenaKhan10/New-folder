import pandas as pd
import numpy as np

# 1. Calculate the correlation matrix for a given DataFrame.
# Data: A DataFrame with numerical columns.
def calculate_correlation_matrix():
   df = pd.DataFrame(np.random.rand(100, 10), columns=list('ABCDEFGHIJ'))
   print(df.corr())
      

# 2. Normalize the data in a DataFrame.
# Data: A DataFrame with numerical columns.
def normalize_data():
    df = pd.DataFrame(np.random.rand(100, 10), columns=list('ABCDEFGHIJ'))
    
    # Using the min max feature scaling
    for column in df.columns:
        df[column] = (df[column]-df[column].min()) / (df[column].max()-df[column].min())
        
    # using the maximum absolute scaling
    for column in df.columns:
        df[column] = df[column] / df[column].abs().max()
        
    # using z score method
    for column in df.columns:
        df[column] = (df[column]-df[column].mean()) / (df[column].std())
        
        
# 3. Handle missing data by filling with the mean of the column.
# Data: A DataFrame with some missing values.
def replace_none_value_with_mean():
    df = pd.DataFrame(np.random.rand(100, 10), columns=list('ABCDEFGHIJ'))
    df.loc[::10, 'A'] = np.nan
    df.fillna(df.mean(), inplace=True)


# 4. Merge two DataFrames on a common column.
# Data: Two DataFrames with at least one common column.
def merge_dataframes():
    df1 = pd.DataFrame({'key': ['A', 'B', 'C', 'D'], 'value': np.random.rand(4)})
    df2 = pd.DataFrame({'key': ['B', 'D', 'E', 'F'], 'value': np.random.rand(4)})
    df = pd.merge(df1, df2, on='key', how='outer', suffixes=['_df1', '_df2'])
    print(df)
    
# merge_dataframes()


# 5. Pivot a DataFrame to create a summary table.
# Data: A DataFrame with multiple columns.
def pivot_dataframe():
    df = pd.DataFrame({'A': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'],
                    'B': ['one', 'one', 'two', 'two', 'one', 'one'],
                    'C': np.random.rand(6)})
    df = df.pivot_table(df, index=['A'])
    print(df)
    

# 6. Create a multi-index DataFrame.
# Data: A DataFrame with hierarchical indexing.
def create_multi_index():
    arrays = [np.array(['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux']),
            np.array(['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two'])]
    df = pd.DataFrame(np.random.rand(8, 4), index=arrays, columns=list('ABCD'))
    
    print(df)
    
# create_multi_index()

# 7. Group data by multiple columns and calculate aggregate statistics.
# Data: A DataFrame with multiple columns.
def group_by_and_aggregation():
    df = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'],
                    'B': ['one', 'one', 'two', 'two', 'one', 'one', 'two', 'two'],
                    'C': np.random.rand(8),
                    'D': np.random.rand(8)})
    
    aggregated = df.groupby(['A','B']).aggregate({'C':'mean', 'D':'count'})

    print(df)
    print(aggregated)
    
# group_by_and_aggregation()


# 8. Filter rows based on a condition applied to multiple columns.
# Data: A DataFrame with multiple columns.
def filter_row_based_on_conditions():
    df = pd.DataFrame({'A': np.random.rand(10), 'B': np.random.rand(10)})
    
    filtered = df[(df['A'] > 0.5) & (df['B'] < 0.5)]
    
    print(df)
    print(filtered)
    
# filter_row_based_on_conditions()

# 9. Create a rolling window calculation.
# Data: A time series DataFrame.
def rolling_window_calculation():
    df = pd.DataFrame({'A': np.random.rand(100)}, index=pd.date_range('20210101', periods=100))
    print(df)
    
rolling_window_calculation()
    

# 10. Resample time series data to a different frequency.
# Data: A time series DataFrame.
# df = pd.DataFrame({'A': np.random.rand(100)}, index=pd.date_range('20210101', periods=100))

# 11. Calculate the cumulative sum of a column.
# Data: A DataFrame with numerical columns.
# df = pd.DataFrame(np.random.rand(100, 10), columns=list('ABCDEFGHIJ'))

# 12. Create a custom function to apply to each row of a DataFrame.
# Data: A DataFrame with multiple columns.
# df = pd.DataFrame(np.random.rand(100, 10), columns=list('ABCDEFGHIJ'))

# 13. Calculate the rank of values within a group.
# Data: A DataFrame with multiple columns.
# df = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'],
#                    'B': np.random.rand(8)})

# 14. Create a lagged version of a time series column.
# Data: A time series DataFrame.
# df = pd.DataFrame({'A': np.random.rand(100)}, index=pd.date_range('20210101', periods=100))

# 15. Calculate the exponentially weighted moving average.
# Data: A time series DataFrame.
# df = pd.DataFrame({'A': np.random.rand(100)}, index=pd.date_range('20210101', periods=100))

# 16. Perform a left join on two DataFrames.
# Data: Two DataFrames with at least one common column.
# df1 = pd.DataFrame({'key': ['A', 'B', 'C', 'D'], 'value': np.random.rand(4)})
# df2 = pd.DataFrame({'key': ['B', 'D', 'E', 'F'], 'value': np.random.rand(4)})

# 17. Create a scatter plot from a DataFrame.
# Data: A DataFrame with numerical columns.
# df = pd.DataFrame(np.random.rand(100, 10), columns=list('ABCDEFGHIJ'))

# 18. Calculate the percentage change of a column.
# Data: A DataFrame with numerical columns.
# df = pd.DataFrame(np.random.rand(100, 10), columns=list('ABCDEFGHIJ'))

# 19. Create a heatmap of the correlation matrix.
# Data: A DataFrame with numerical columns.
# df = pd.DataFrame(np.random.rand(100, 10), columns=list('ABCDEFGHIJ'))

# 20. Perform a principal component analysis (PCA) on a DataFrame.
# Data: A DataFrame with numerical columns.
# df = pd.DataFrame(np.random.rand(100, 10), columns=list('ABCDEFGHIJ'))

# 21. Create a box plot for each column in a DataFrame.
# Data: A DataFrame with numerical columns.
# df = pd.DataFrame(np.random.rand(100, 10), columns=list('ABCDEFGHIJ'))

# 22. Calculate the z-score for each value in a column.
# Data: A DataFrame with numerical columns.
# df = pd.DataFrame(np.random.rand(100, 10), columns=list('ABCDEFGHIJ'))

# 23. Create a histogram for each column in a DataFrame.
# Data: A DataFrame with numerical columns.
# df = pd.DataFrame(np.random.rand(100, 10), columns=list('ABCDEFGHIJ'))

# 24. Perform a t-test between two columns.
# Data: A DataFrame with at least two numerical columns.
# df = pd.DataFrame(np.random.rand(100, 10), columns=list('ABCDEFGHIJ'))

# 25. Create a violin plot for each column in a DataFrame.
# Data: A DataFrame with numerical columns.
# df = pd.DataFrame(np.random.rand(100, 10), columns=list('ABCDEFGHIJ'))

# 26. Calculate the rolling standard deviation of a column.
# Data: A time series DataFrame.
# df = pd.DataFrame({'A': np.random.rand(100)}, index=pd.date_range('20210101', periods=100))

# 27. Create a pair plot for a DataFrame.
# Data: A DataFrame with numerical columns.
# df = pd.DataFrame(np.random.rand(100, 10), columns=list('ABCDEFGHIJ'))

# 28. Perform a chi-square test of independence between two columns.
# Data: A DataFrame with at least two categorical columns.
# df = pd.DataFrame({'A': np.random.choice(['foo', 'bar'], 100), 'B': np.random.choice(['one', 'two'], 100)})

# 29. Create a bar plot for each column in a DataFrame.
# Data: A DataFrame with categorical columns.
# df = pd.DataFrame({'A': np.random.choice(['foo', 'bar'], 100), 'B': np.random.choice(['one', 'two'], 100)})

# 30. Calculate the mean absolute deviation of a column.
# Data: A DataFrame with numerical columns.
# df = pd.DataFrame(np.random.rand(100, 10), columns=list('ABCDEFGHIJ'))

# 31. Create a DataFrame from a dictionary of lists.
# Data: A dictionary with lists of equal length.
# data = {'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8], 'C': [9, 10, 11, 12]}

# 32. Create a DataFrame from a list of dictionaries.
# Data: A list of dictionaries with the same keys.
# data = [{'A': 1, 'B': 2}, {'A': 3, 'B': 4}, {'A': 5, 'B': 6}]

# 33. Create a DataFrame from a NumPy array.
# Data: A NumPy array with shape (n, m).
# data = np.random.rand(100, 10)

# 34. Create a DataFrame from a CSV file.
# Data: A CSV file with a header row.
# df = pd.read_csv('data.csv')

# 35. Create a DataFrame from an Excel file.
# Data: An Excel file with a header row.
# df = pd.read_excel('data.xlsx')

# 36. Create a DataFrame from a SQL query.
# Data: A SQL query that returns a result set.
# import sqlite3
# conn = sqlite3.connect('database.db')
# df = pd.read_sql_query('SELECT * FROM table', conn)

# 37. Create a DataFrame from a JSON file.
# Data: A JSON file with an array of objects.
# df = pd.read_json('data.json')

# 38. Create a DataFrame from an HTML table.
# Data: An HTML file with a table element.
# df = pd.read_html('data.html')[0]

# 39. Create a DataFrame from a dictionary of Series.
# Data: A dictionary with Series of equal length.
# data = {'A': pd.Series([1, 2, 3, 4]), 'B': pd.Series([5, 6, 7, 8])}

# 40. Create a DataFrame from a list of tuples.
# Data: A list of tuples with the same length.
# data = [(1, 2), (3, 4), (5, 6)]

# 41. Create a DataFrame from a list of lists.
# Data: A list of lists with the same length.
# data = [[1, 2], [3, 4], [5, 6]]

# 42. Create a DataFrame from a dictionary of NumPy arrays.
# Data: A dictionary with NumPy arrays of equal length.
# data = {'A': np.array([1, 2, 3, 4]), 'B': np.array([5, 6, 7, 8])}

# 43. Create a DataFrame from a dictionary of DataFrames.
# Data: A dictionary with DataFrames of equal length.
# data = {'A': pd.DataFrame(np.random.rand(4, 2)), 'B': pd.DataFrame(np.random.rand(4, 2))}

# 44. Create a DataFrame from a list of namedtuples.
# Data: A list of namedtuples with the same fields.
# from collections import namedtuple
# Point = namedtuple('Point', ['x', 'y'])
# data = [Point(1, 2), Point(3, 4), Point(5, 6)]

# 45. Create a DataFrame from a list of records.
# Data: A list of records with the same fields.
# data = [{'A': 1, 'B': 2}, {'A': 3, 'B': 4}, {'A': 5, 'B': 6}]

# 46. Create a DataFrame from a list of dictionaries with missing keys.
# Data: A list of dictionaries with some missing keys.
# data = [{'A': 1, 'B': 2}, {'A': 3}, {'B': 4}]

# 47. Create a DataFrame from a list of lists with missing values.
# Data: A list of lists with some missing values.
# data = [[1, 2], [3, None], [5, 6]]

# 48. Create a DataFrame from a list of lists with different lengths.
# Data: A list of lists with different lengths.
# data = [[1, 2], [3, 4, 5], [6]]

# 49. Create a DataFrame from a list of lists with different types.
# Data: A list of lists with different types.
# data = [[1, 'a'], [2, 'b'], [3, 'c']]

# 50. Create a DataFrame from a list of lists with different types and missing values.
# Data: A list of lists with different types and some missing values.
# data = [[1, 'a'], [2, None], [3, 'c']]

# 51. Create a DataFrame from a list of lists with different types and different lengths.
# Data: A list of lists with different types and different lengths.
# data = [[1, 'a'], [2, 'b', 3], [4]]

# 52. Create a DataFrame from a list of lists with different types, different lengths, and missing values.
# Data: A list of lists with different types, different lengths, and some missing values.
# data = [[1, 'a'], [2, None, 3], [4]]

# 53. Create a DataFrame from a list of lists with different types, different lengths, missing values, and nested lists.
# Data: A list of lists with different types, different lengths, some missing values, and nested lists.
# data = [[1, 'a'], [2, None, [3, 4]], [5]]

# 54. Create a DataFrame from a list of lists with different types, different lengths, missing values, nested lists, and dictionaries.
# Data: A list of lists with different types, different lengths, some missing values, nested lists, and dictionaries.
# data = [[1, 'a'], [2, None, [3, 4], {'x': 5}], [6]]

# 55. Create a DataFrame from a list of lists with different types, different lengths, missing values, nested lists, dictionaries, and tuples.
# Data: A list of lists with different types, different lengths, some missing values, nested lists, dictionaries, and tuples.
# data = [[1, 'a'], [2, None, [3, 4], {'x': 5}, (6, 7)], [8]]

# 56. Create a DataFrame from a list of lists with different types, different lengths, missing values, nested lists, dictionaries, tuples, and namedtuples.
# Data: A list of lists with different types, different lengths, some missing values, nested lists, dictionaries, tuples, and namedtuples.
# from collections import namedtuple
# Point = namedtuple('Point', ['x', 'y'])
# data = [[1, 'a'], [2, None, [3, 4], {'x': 5}, (6, 7), Point(8, 9)], [10]]

# 57. Create a DataFrame from a list of lists with different types, different lengths, missing values, nested lists, dictionaries, tuples, namedtuples, and sets.
# Data: A list of lists with different types, different lengths, some missing values, nested lists, dictionaries, tuples, namedtuples, and sets.
# from collections import namedtuple
# Point = namedtuple('Point', ['x', 'y'])
# data = [[1, 'a'], [2, None, [3, 4], {'x': 5}, (6, 7), Point(8, 9), {10, 11}], [12]]

# 58. Create a DataFrame from a list of lists with different types, different lengths, missing values, nested lists, dictionaries, tuples, namedtuples, sets, and frozensets.
# Data: A list of lists with different types, different lengths, some missing values, nested lists, dictionaries, tuples, namedtuples, sets, and frozensets.
# from collections import namedtuple
# Point = namedtuple('Point', ['x', 'y'])
# data = [[1, 'a'], [2, None, [3, 4], {'x': 5}, (6, 7), Point(8, 9), {10, 11}, frozenset([12, 13])], [14]]

# 59. Create a DataFrame from a list of lists with different types, different lengths, missing values, nested lists, dictionaries, tuples, namedtuples, sets, frozensets, and arrays.
# Data: A list of lists with different types, different lengths, some missing values, nested lists, dictionaries, tuples, namedtuples, sets, frozensets, and arrays.
# from collections import namedtuple
# Point = namedtuple('Point', ['x', 'y'])
# data = [[1, 'a'], [2, None, [3, 4], {'x': 5}, (6, 7), Point(8, 9), {10, 11}, frozenset([12, 13]), np.array([14, 15])], [16]]

# 60. Create a DataFrame from a list of lists with different types, different lengths, missing values, nested lists, dictionaries, tuples, namedtuples, sets, frozensets, arrays, and DataFrames.
# Data: A list of lists with different types, different lengths, some missing values, nested lists, dictionaries, tuples, namedtuples, sets, frozensets, arrays, and DataFrames.
# from collections import namedtuple
# Point = namedtuple('Point', ['x', 'y'])
# data = [[1, 'a'], [2, None, [3, 4], {'x': 5}, (6, 7), Point(8, 9), {10, 11}, frozenset([12, 13]), np.array([14, 15]), pd.DataFrame(np.random.rand(2, 2))], [16]]

# 61. Create a DataFrame from a list of lists with different types, different lengths, missing values, nested lists, dictionaries, tuples, namedtuples, sets, frozensets, arrays, DataFrames, and Series.
# Data: A list of lists with different types, different lengths, some missing values, nested lists, dictionaries, tuples, namedtuples, sets, frozensets, arrays, DataFrames, and Series.
# from collections import namedtuple
# Point = namedtuple('Point', ['x', 'y'])
# data = [[1, 'a'], [2, None, [3, 4], {'x': 5}, (6, 7), Point(8, 9), {10, 11}, frozenset([12, 13]), np.array([14, 15]), pd.DataFrame(np.random.rand(2, 2)), pd.Series([16, 17])], [18]]

# 62. Calculate the mean of each column in a DataFrame.
# Data: A DataFrame with numerical columns.
def calculate_mean():
    df = pd.DataFrame(np.random.rand(100, 10), columns=list('ABCDEFGHIJ'))
    print(df.mean())

# 63. Calculate the median of each column in a DataFrame.
# Data: A DataFrame with numerical columns.
def calculate_median():
    df = pd.DataFrame(np.random.rand(100, 10), columns=list('ABCDEFGHIJ'))
    print(df.median())

# 64. Calculate the mode of each column in a DataFrame.
# Data: A DataFrame with numerical columns.
def calculate_mode():
    df = pd.DataFrame(np.random.rand(100, 10), columns=list('ABCDEFGHIJ'))
    print(df.mode())

# 65. Calculate the variance of each column in a DataFrame.
# Data: A DataFrame with numerical columns.
def calculate_variance():
    df = pd.DataFrame(np.random.rand(100, 10), columns=list('ABCDEFGHIJ'))
    print(df.var())

# 66. Calculate the standard deviation of each column in a DataFrame.
# Data: A DataFrame with numerical columns.
def calculate_std():
    df = pd.DataFrame(np.random.rand(100, 10), columns=list('ABCDEFGHIJ'))
    print(df.std())

# 67. Calculate the skewness of each column in a DataFrame.
# Data: A DataFrame with numerical columns.
def calculate_skewness():
    df = pd.DataFrame(np.random.rand(100, 10), columns=list('ABCDEFGHIJ'))
    print(df.skew())

# 68. Calculate the kurtosis of each column in a DataFrame.
# Data: A DataFrame with numerical columns.
def calculate_kurtosis():
    df = pd.DataFrame(np.random.rand(100, 10), columns=list('ABCDEFGHIJ'))
    print(df.kurt())

# 69. Calculate the minimum value of each column in a DataFrame.
# Data: A DataFrame with numerical columns.
def calculate_min():
    df = pd.DataFrame(np.random.rand(100, 10), columns=list('ABCDEFGHIJ'))
    print(df.min())

# 70. Calculate the maximum value of each column in a DataFrame.
# Data: A DataFrame with numerical columns.
def calculate_max():
    df = pd.DataFrame(np.random.rand(100, 10), columns=list('ABCDEFGHIJ'))
    print(df.max())

# 71. Calculate the range of each column in a DataFrame.
# Data: A DataFrame with numerical columns.
def calculate_range():
    df = pd.DataFrame(np.random.rand(100, 10), columns=list('ABCDEFGHIJ'))
    print(df.max() - df.min())

# 72. Calculate the interquartile range (IQR) of each column in a DataFrame.
# Data: A DataFrame with numerical columns.
def calculate_iqr():
    df = pd.DataFrame(np.random.rand(100, 10), columns=list('ABCDEFGHIJ'))
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    print(IQR)

# 73. Calculate the 10th percentile of each column in a DataFrame.
# Data: A DataFrame with numerical columns.
def calculate_10th_percentile():
    df = pd.DataFrame(np.random.rand(100, 10), columns=list('ABCDEFGHIJ'))
    print(df.quantile(0.1))

# 74. Calculate the 90th percentile of each column in a DataFrame.
# Data: A DataFrame with numerical columns.
def calculate_90th_percentile():
    df = pd.DataFrame(np.random.rand(100, 10), columns=list('ABCDEFGHIJ'))
    print(df.quantile(0.9))

# 75. Calculate the correlation between two columns in a DataFrame.
# Data: A DataFrame with numerical columns.
def calculate_correlation():
    df = pd.DataFrame(np.random.rand(100, 10), columns=list('ABCDEFGHIJ'))
    print(df['A'].corr(df['B']))

# 76. Calculate the covariance between two columns in a DataFrame.
# Data: A DataFrame with numerical columns.
def calculate_covariance():
    df = pd.DataFrame(np.random.rand(100, 10), columns=list('ABCDEFGHIJ'))
    print(df['A'].cov(df['B']))

# 77. Calculate the rolling mean of a column in a DataFrame.
# Data: A time series DataFrame.
def calculate_rolling_mean():
    df = pd.DataFrame({'A': np.random.rand(100)}, index=pd.date_range('20210101', periods=100))
    print(df['A'].rolling(window=5).mean())

# 78. Calculate the rolling median of a column in a DataFrame.
# Data: A time series DataFrame.
def calculate_rolling_median():
    df = pd.DataFrame({'A': np.random.rand(100)}, index=pd.date_range('20210101', periods=100))
    print(df['A'].rolling(window=5).median())

# 79. Calculate the rolling variance of a column in a DataFrame.
# Data: A time series DataFrame.
def calculate_rolling_variance():
    df = pd.DataFrame({'A': np.random.rand(100)}, index=pd.date_range('20210101', periods=100))
    print(df['A'].rolling(window=5).var())

# 80. Calculate the rolling standard deviation of a column in a DataFrame.
# Data: A time series DataFrame.
def calculate_rolling_std():
    df = pd.DataFrame({'A': np.random.rand(100)}, index=pd.date_range('20210101', periods=100))
    print(df['A'].rolling(window=5).std())

# 81. Calculate the rolling skewness of a column in a DataFrame.
# Data: A time series DataFrame.
def calculate_rolling_skewness():
    df = pd.DataFrame({'A': np.random.rand(100)}, index=pd.date_range('20210101', periods=100))
    print(df['A'].rolling(window=5).skew())

# 82. Calculate the rolling kurtosis of a column in a DataFrame.
# Data: A time series DataFrame.
def calculate_rolling_kurtosis():
    df = pd.DataFrame({'A': np.random.rand(100)}, index=pd.date_range('20210101', periods=100))
    print(df['A'].rolling(window=5).kurt())

# 83. Calculate the rolling minimum of a column in a DataFrame.
# Data: A time series DataFrame.
def calculate_rolling_min():
    df = pd.DataFrame({'A': np.random.rand(100)}, index=pd.date_range('20210101', periods=100))
    print(df['A'].rolling(window=5).min())

# 84. Calculate the rolling maximum of a column in a DataFrame.
# Data: A time series DataFrame.
def calculate_rolling_max():
    df = pd.DataFrame({'A': np.random.rand(100)}, index=pd.date_range('20210101', periods=100))
    print(df['A'].rolling(window=5).max())

# 85. Calculate the rolling range of a column in a DataFrame.
# Data: A time series DataFrame.
def calculate_rolling_range():
    df = pd.DataFrame({'A': np.random.rand(100)}, index=pd.date_range('20210101', periods=100))
    print(df['A'].rolling(window=5).apply(lambda x: x.max() - x.min()))

# 86. Calculate the rolling interquartile range (IQR) of a column in a DataFrame.
# Data: A time series DataFrame.
def calculate_rolling_iqr():
    df = pd.DataFrame({'A': np.random.rand(100)}, index=pd.date_range('20210101', periods=100))
    print(df['A'].rolling(window=5).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25)))

# 87. Calculate the rolling 10th percentile of a column in a DataFrame.
# Data: A time series DataFrame.
def calculate_rolling_10th_percentile():
    df = pd.DataFrame({'A': np.random.rand(100)}, index=pd.date_range('20210101', periods=100))
    print(df['A'].rolling(window=5).apply(lambda x: np.percentile(x, 10)))

# 88. Calculate the rolling 90th percentile of a column in a DataFrame.
# Data: A time series DataFrame.
def calculate_rolling_90th_percentile():
    df = pd.DataFrame({'A': np.random.rand(100)}, index=pd.date_range('20210101', periods=100))
    print(df['A'].rolling(window=5).apply(lambda x: np.percentile(x, 90)))

# 89. Calculate the rolling correlation between two columns in a DataFrame.
# Data: A time series DataFrame.
def calculate_rolling_correlation():
    df = pd.DataFrame({'A': np.random.rand(100), 'B': np.random.rand(100)}, index=pd.date_range('20210101', periods=100))
    print(df['A'].rolling(window=5).corr(df['B']))

# 90. Calculate the rolling covariance between two columns in a DataFrame.
# Data: A time series DataFrame.
def calculate_rolling_covariance():
    df = pd.DataFrame({'A': np.random.rand(100), 'B': np.random.rand(100)}, index=pd.date_range('20210101', periods=100))
    print(df['A'].rolling(window=5).cov(df['B']))

# 91. Calculate the expanding mean of a column in a DataFrame.
# Data: A time series DataFrame.
def calculate_expanding_mean():
    df = pd.DataFrame({'A': np.random.rand(100)}, index=pd.date_range('20210101', periods=100))
    print(df['A'].expanding().mean())

# 92. Calculate the expanding median of a column in a DataFrame.
# Data: A time series DataFrame.
def calculate_expanding_median():
    df = pd.DataFrame({'A': np.random.rand(100)}, index=pd.date_range('20210101', periods=100))
    print(df['A'].expanding().median())

# 93. Calculate the expanding variance of a column in a DataFrame.
# Data: A time series DataFrame.
def calculate_expanding_variance():
    df = pd.DataFrame({'A': np.random.rand(100)}, index=pd.date_range('20210101', periods=100))
    print(df['A'].expanding().var())

# 94. Calculate the expanding standard deviation of a column in a DataFrame.
# Data: A time series DataFrame.
def calculate_expanding_std():
    df = pd.DataFrame({'A': np.random.rand(100)}, index=pd.date_range('20210101', periods=100))
    print(df['A'].expanding().std())

# 95. Calculate the expanding skewness of a column in a DataFrame.
# Data: A time series DataFrame.
def calculate_expanding_skewness():
    df = pd.DataFrame({'A': np.random.rand(100)}, index=pd.date_range('20210101', periods=100))
    print(df['A'].expanding().skew())

# 96. Calculate the expanding kurtosis of a column in a DataFrame.
# Data: A time series DataFrame.
def calculate_expanding_kurtosis():
    df = pd.DataFrame({'A': np.random.rand(100)}, index=pd.date_range('20210101', periods=100))
    print(df['A'].expanding().kurt())

# 97. Calculate the expanding minimum of a column in a DataFrame.
# Data: A time series DataFrame.
def calculate_expanding_min():
    df = pd.DataFrame({'A': np.random.rand(100)}, index=pd.date_range('20210101', periods=100))
    print(df['A'].expanding().min())

# 98. Calculate the expanding maximum of a column in a DataFrame.
# Data: A time series DataFrame.
def calculate_expanding_max():
    df = pd.DataFrame({'A': np.random.rand(100)}, index=pd.date_range('20210101', periods=100))
    print(df['A'].expanding().max())

# 99. Calculate the expanding range of a column in a DataFrame.
# Data: A time series DataFrame.
def calculate_expanding_range():
    df = pd.DataFrame({'A': np.random.rand(100)}, index=pd.date_range('20210101', periods=100))
    print(df['A'].expanding().apply(lambda x: x.max() - x.min()))

# 100. Calculate the expanding interquartile range (IQR) of a column in a DataFrame.
# Data: A time series DataFrame.
def calculate_expanding_iqr():
    df = pd.DataFrame({'A': np.random.rand(100)}, index=pd.date_range('20210101', periods=100))
    print(df['A'].expanding().apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25)))

# 101. Calculate the expanding 10th percentile of a column in a DataFrame.
# Data: A time series DataFrame.
def calculate_expanding_10th_percentile():
    df = pd.DataFrame({'A': np.random.rand(100)}, index=pd.date_range('20210101', periods=100))
    print(df['A'].expanding().apply(lambda x: np.percentile(x, 10)))

# 102. Calculate the expanding 90th percentile of a column in a DataFrame.
# Data: A time series DataFrame.
def calculate_expanding_90th_percentile():
    df = pd.DataFrame({'A': np.random.rand(100)}, index=pd.date_range('20210101', periods=100))
    print(df['A'].expanding().apply(lambda x: np.percentile(x, 90)))

# 103. Calculate the expanding correlation between two columns in a DataFrame.
# Data: A time series DataFrame.
def calculate_expanding_correlation():
    df = pd.DataFrame({'A': np.random.rand(100), 'B': np.random.rand(100)}, index=pd.date_range('20210101', periods=100))
    print(df['A'].expanding().corr(df['B']))

# 104. Calculate the expanding covariance between two columns in a DataFrame.
# Data: A time series DataFrame.
def calculate_expanding_covariance():
    df = pd.DataFrame({'A': np.random.rand(100), 'B': np.random.rand(100)}, index=pd.date_range('20210101', periods=100))
    print(df['A'].expanding().cov(df['B']))

# 105. Calculate the exponentially weighted mean of a column in a DataFrame.
# Data: A time series DataFrame.
def calculate_ewm_mean():
    df = pd.DataFrame({'A': np.random.rand(100)}, index=pd.date_range('20210101', periods=100))
    print(df['A'].ewm(span=5).mean())

# 106. Calculate the exponentially weighted variance of a column in a DataFrame.
# Data: A time series DataFrame.
def calculate_ewm_variance():
    df = pd.DataFrame({'A': np.random.rand(100)}, index=pd.date_range('20210101', periods=100))
    print(df['A'].ewm(span=5).var())

# 107. Calculate the exponentially weighted standard deviation of a column in a DataFrame.
# Data: A time series DataFrame.
def calculate_ewm_std():
    df = pd.DataFrame({'A': np.random.rand(100)}, index=pd.date_range('20210101', periods=100))
    print(df['A'].ewm(span=5).std())

# 108. Calculate the exponentially weighted correlation between two columns in a DataFrame.
# Data: A time series DataFrame.
def calculate_ewm_correlation():
    df = pd.DataFrame({'A': np.random.rand(100), 'B': np.random.rand(100)}, index=pd.date_range('20210101', periods=100))
    print(df['A'].ewm(span=5).corr(df['B']))

# 109. Calculate the exponentially weighted covariance between two columns in a DataFrame.
# Data: A time series DataFrame.
def calculate_ewm_covariance():
    df = pd.DataFrame({'A': np.random.rand(100), 'B': np.random.rand(100)}, index=pd.date_range('20210101', periods=100))
    print(df['A'].ewm(span=5).cov(df['B']))

# 110. Calculate the exponentially weighted skewness of a column in a DataFrame.
# Data: A time series DataFrame.
def calculate_ewm_skewness():
    df = pd.DataFrame({'A': np.random.rand(100)}, index=pd.date_range('20210101', periods=100))
    print(df['A'].ewm(span=5).apply(lambda x: x.skew()))

# 111. Calculate the exponentially weighted kurtosis of a column in a DataFrame.
# Data: A time series DataFrame.
def calculate_ewm_kurtosis():
    df = pd.DataFrame({'A': np.random.rand(100)}, index=pd.date_range('20210101', periods=100))
    print(df['A'].ewm(span=5).apply(lambda x: x.kurt()))

# 112. Calculate the exponentially weighted minimum of a column in a DataFrame.
# Data: A time series DataFrame.
def calculate_ewm_min():
    df = pd.DataFrame({'A': np.random.rand(100)}, index=pd.date_range('20210101', periods=100))
    print(df['A'].ewm(span=5).apply(lambda x: x.min()))

# 113. Calculate the exponentially weighted maximum of a column in a DataFrame.
# Data: A time series DataFrame.
def calculate_ewm_max():
    df = pd.DataFrame({'A': np.random.rand(100)}, index=pd.date_range('20210101', periods=100))
    print(df['A'].ewm(span=5).apply(lambda x: x.max()))

# 114. Calculate the exponentially weighted range of a column in a DataFrame.
# Data: A time series DataFrame.
def calculate_ewm_range():
    df = pd.DataFrame({'A': np.random.rand(100)}, index=pd.date_range('20210101', periods=100))
    print(df['A'].ewm(span=5).apply(lambda x: x.max() - x.min()))