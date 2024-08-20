# 1. Calculate the correlation matrix for a given DataFrame.
# Data: A DataFrame with numerical columns.
# df = pd.DataFrame(np.random.rand(100, 10), columns=list('ABCDEFGHIJ'))

# 2. Normalize the data in a DataFrame.
# Data: A DataFrame with numerical columns.
# df = pd.DataFrame(np.random.rand(100, 10), columns=list('ABCDEFGHIJ'))

# 3. Handle missing data by filling with the mean of the column.
# Data: A DataFrame with some missing values.
# df = pd.DataFrame(np.random.rand(100, 10), columns=list('ABCDEFGHIJ'))
# df.loc[::10, 'A'] = np.nan

# 4. Merge two DataFrames on a common column.
# Data: Two DataFrames with at least one common column.
# df1 = pd.DataFrame({'key': ['A', 'B', 'C', 'D'], 'value': np.random.rand(4)})
# df2 = pd.DataFrame({'key': ['B', 'D', 'E', 'F'], 'value': np.random.rand(4)})

# 5. Pivot a DataFrame to create a summary table.
# Data: A DataFrame with multiple columns.
# df = pd.DataFrame({'A': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'],
#                    'B': ['one', 'one', 'two', 'two', 'one', 'one'],
#                    'C': np.random.rand(6)})

# 6. Create a multi-index DataFrame.
# Data: A DataFrame with hierarchical indexing.
# arrays = [np.array(['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux']),
#           np.array(['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two'])]
# df = pd.DataFrame(np.random.rand(8, 4), index=arrays, columns=list('ABCD'))

# 7. Group data by multiple columns and calculate aggregate statistics.
# Data: A DataFrame with multiple columns.
# df = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'],
#                    'B': ['one', 'one', 'two', 'two', 'one', 'one', 'two', 'two'],
#                    'C': np.random.rand(8),
#                    'D': np.random.rand(8)})

# 8. Filter rows based on a condition applied to multiple columns.
# Data: A DataFrame with multiple columns.
# df = pd.DataFrame({'A': np.random.rand(10), 'B': np.random.rand(10)})

# 9. Create a rolling window calculation.
# Data: A time series DataFrame.
# df = pd.DataFrame({'A': np.random.rand(100)}, index=pd.date_range('20210101', periods=100))

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



