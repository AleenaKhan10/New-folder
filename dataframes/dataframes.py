import pandas as pd
import numpy as np

exam_data = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
                'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
                'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
                'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

df = pd.DataFrame(exam_data, index=labels)

# Write a Pandas program to create and display a DataFrame from a specified dictionary data which has the index labels.

# Sample DataFrame:
# exam_data = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
# 'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
# 'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
# 'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
# labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']


def create_df_from_dict_with_labels():
    
    print(df)
    

# create_df_from_dict_with_labels()


def display_summary_of_dataframe():

    print(df.info())
    
# display_summary_of_dataframe()


def get_first_three_column():

    print(df.iloc[:3])
    
# get_first_three_column()


def get_column_with_name():
    
    print(df[['score', 'name']])

# get_column_with_name()


def get_specific_rows_and_columns():
    
    print(df.iloc[[1,3,5,6], [1,2]])
    
# get_specific_rows_and_columns()

def get_row_where_attempts_greater_than_2():
    
    print(df[df["attempts"] > 2])
    
# get_row_where_attempts_greater_than_2()


def get_count_of_rows_colunms():
    
    print(f"Number of rows : {len(df.axes[0])}")
    print(f"Number of columns : {len(df.axes[1])}")
    
# get_count_of_rows_colunms()

def select_rows_with_missing_score():

    print(df[df['score'].isnull()])
   
# select_rows_with_missing_score()
 

def select_rows_within_15_20_range():
    
    print(df['score'].between(15, 20))
    
# select_rows_within_15_20_range()


def select_rows_where_attempts_less_than_2_and_score_greater_than_15():
    
    print(df[(df['attempts'] < 2) & (df['score'] > 15)])
    
# select_rows_where_attempts_less_than_2_and_score_greater_than_15()


def update_row_d_column_score_to_115():

    df.loc['d', 'score'] = 11.5    
    print(df)
     
# update_row_d_column_score_to_115()


def get_sum_of_attempts():
    
    print(df['attempts'].sum())

# get_sum_of_attempts()


def get_mean_of_score():
    
    print(df['score'].mean())

# get_mean_of_score()


def append_and_delete_row_k(df):
    df.loc['k'] = [1, 'Suresh', 'yes', 15.5]
    print(df)
    df = df.drop('k')
    print(df)
    
# append_and_delete_row_k(df)


def sort_values_by_name_than_by_score(df : pd.DataFrame):
    df = df.sort_values(by=['name', 'score'], ascending=[False, True])
    print(df)
    
# sort_values_by_name_than_by_score(df)

def change_yes_to_true_and_no_to_false(df: pd.DataFrame):
    df['qualify'] = df['qualify'].map({'yes': True, 'no':False})
    print(df)
    
# change_yes_to_true_and_no_to_false(df)


def update_name_james_to_suresh(df: pd.DataFrame):
    df.loc[df['name'] == 'James', 'name'] = 'Suresh'
    print(df)
    
# update_name_james_to_suresh(df)


def delete_column_attempts(df : pd.DataFrame):
    df.pop('attempts')
    print(df)
    
# delete_column_attempts(df)


def insert_new_column(df: pd.DataFrame):
    df['color'] = ['Red','Blue','Orange','Red','White','White','Blue','Green','Green','Red']
    print(df)
    
# insert_new_column(df)


def iterate_over_rows():
    exam_data = [{'name':'Anastasia', 'score':12.5}, {'name':'Dima','score':9}, {'name':'Katherine','score':16.5}]
    df = pd.DataFrame(exam_data)
    
    for index, row in df.iterrows():
        print(row['name'], row['score'])
        
# iterate_over_rows()


def get_column_headers_as_list(df : pd.DataFrame):
    
    print(list(df.columns.values))
    
# get_column_headers_as_list(df)


def update_column_values(df : pd.DataFrame):
    
    df.columns = ['col1', 'col2', 'col3', 'col4']
    print(df)
    
# update_column_values(df)


def select_rows_based_on_values(df: pd.DataFrame):
    print(df.loc[df['score'] > 9])
    
# select_rows_based_on_values(df)


def change_order_of_columns(df: pd.DataFrame):
    print(df)
    df = df[['score', 'qualify', 'name', 'attempts']]
    print(df)
    
# change_order_of_columns(df)


def add_row_to_dataframe(df: pd.DataFrame):
    df.loc['k'] = ['albert', 20, 'no',  10]
    print(df)
    
# add_row_to_dataframe(df)


def dataframe_to_csv(df: pd.DataFrame):
    df.to_csv('csv_file.csv', sep='\t', index=False)
    dfc = pd.read_csv('csv_file.csv')
    print(dfc)
    
# dataframe_to_csv(df)
    
def count_city_wise_number_of_people(df: pd.DataFrame):
    df = pd.DataFrame({'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
    'city': ['California', 'Los Angeles', 'California', 'California', 'California', 'Los Angeles', 'Los Angeles', 'Georgia', 'Georgia', 'Los Angeles']})
    print(df)
    df1 = df.groupby(['city']).size().reset_index(name='number of people')
    print(df1)
    
# count_city_wise_number_of_people(df)


def replace_nan_to_zero(df: pd.DataFrame):
    print(df)
    df = df.fillna(0)
    print(df)
    
# replace_nan_to_zero(df)


def replace_nan_to_zero(df: pd.DataFrame):
    print(df)
    df.reset_index(level=0, inplace=True)
    print(df)
    df = df.to_string(index=False)
    print(df)
    
# replace_nan_to_zero(df)


def update_specific_value(df: pd.DataFrame):
    df.iloc[8, 1] = 10.10
    print(df)
    
# update_specific_value(df)


def count_null_values(df: pd.DataFrame):
    print(df.isnull().values.sum())
    
# count_null_values(df)


def combine_two_series_in_dataframe(df: pd.DataFrame):
    s1 = pd.Series(['100', '200', 'python', '300.12', '400'])
    s2 = pd.Series(['10', '20', 'php', '30.12', '40'])
    
    print(pd.concat([s1, s2], axis=1))
    
# combine_two_series_in_dataframe(df)


def convert_datatype_of_colunm(df: pd.DataFrame):
    s = pd.Series(['3/11/2000', '3/12/2000', '3/13/2000'])
    r = pd.to_datetime(pd.Series(s))
    df = pd.DataFrame(r)
    print(df)
    
# convert_datatype_of_colunm(df)



def group_by_first_column_and_get_second_column_in_list(df: pd.DataFrame):
    df = pd.DataFrame( {'col1':['C1','C1','C2','C2','C2','C3','C2'], 'col2':[1,2,3,3,4,6,5]})
    print(df.groupby('col1')['col2'].apply(list))
    
# group_by_first_column_and_get_second_column_in_list(df)