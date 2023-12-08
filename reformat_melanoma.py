# import pandas as pd

# df_coordinates = pd.read_csv('../../melanoma/0_1_coordinates.csv')
# df_coordinates.head()

# # cell_coords = df_coordinates[['uuid','x0','y0']]
# # cell_types = df_coordinates[['uuid','celltype']]

# df_features = pd.read_csv('../../melanoma/0_1_feature.csv')
# df_features.head()

# # concatenate the sample_id and fov_id to create a new column named region_id
# df_features['region_id'] = df_features.apply(lambda row: f"{row['sample_id']}_{row['fov_id']}", axis=1)

# df_merged = pd.merge(df_coordinates, df_features, on='uuid')
# df_merged.head()

# # if df_merged['fov_id_x'].equals(df_merged['fov_id_y']):
# #     # just retain only one of the columns and rename it to sample_id
# #     df_merged = df_merged.drop(['fov_id_x'], axis=1)
    
# #     df_merged = df_merged.rename(columns={'fov_id_y': 'fov_id'})
# #     # pass
# # else:
# #     # if they are not equal, print out where the difference is
# #     print('fov_id_x and fov_id_y are not equal')
    
# # if df_merged['sample_id_x'].equals(df_merged['sample_id_y']):
# #     # just retain only one of the columns and rename it to sample_id
# #     df_merged = df_merged.drop(['sample_id_x'], axis=1)
    
# #     df_merged = df_merged.rename(columns={'sample_id_y': 'sample_id'})
# #     # pass
# # else:
# #     # if they are not equal, print out where the difference is
# #     print('sample_id_x and sample_id_y are not equal')

# # df_merged = df_merged.drop(['celltype_x'], axis=1)
# # df_merged = df_merged.rename(columns={'celltype_y': 'celltype'})

# dfs = {'df_' + region_id: df_merged[df_merged['region_id'] == region_id] for region_id in df_merged['region_id'].unique()}

# # print the dataframes
# for df_name, df in dfs.items():
#     print(f"Dataframe {df_name}:")
#     print(df.head())

# # get the dataframes separated by region_id
# df_0_1_6 = dfs[list(dfs.keys())[0]]
# df_0_1_8 = dfs[list(dfs.keys())[1]]
# df_0_1_9 = dfs[list(dfs.keys())[2]]
# df_0_1_3 = dfs[list(dfs.keys())[3]]
# df_0_1_15 = dfs[list(dfs.keys())[4]]
# df_0_1_16 = dfs[list(dfs.keys())[5]]
# df_0_1_7 = dfs[list(dfs.keys())[6]]
# df_0_1_17 = dfs[list(dfs.keys())[7]]
# df_0_1_1 = dfs[list(dfs.keys())[8]]
# df_0_1_10 = dfs[list(dfs.keys())[9]]
# df_0_1_25 = dfs[list(dfs.keys())[10]]
# df_0_1_4 = dfs[list(dfs.keys())[11]]

# # for all the dataframes, create a new dataframe for the coordinates and the uuid

# df_0_1_6_coord = df_0_1_6[['uuid', 'x0', 'y0']]
# df_0_1_8_coord = df_0_1_8[['uuid', 'x0', 'y0']]
# df_0_1_9_coord = df_0_1_9[['uuid', 'x0', 'y0']]
# df_0_1_3_coord = df_0_1_3[['uuid', 'x0', 'y0']]
# df_0_1_15_coord = df_0_1_15[['uuid', 'x0', 'y0']]
# df_0_1_16_coord = df_0_1_16[['uuid', 'x0', 'y0']]
# df_0_1_7_coord = df_0_1_7[['uuid', 'x0', 'y0']]
# df_0_1_17_coord = df_0_1_17[['uuid', 'x0', 'y0']]
# df_0_1_1_coord = df_0_1_1[['uuid', 'x0', 'y0']]
# df_0_1_10_coord = df_0_1_10[['uuid', 'x0', 'y0']]
# df_0_1_25_coord = df_0_1_25[['uuid', 'x0', 'y0']]
# df_0_1_4_coord = df_0_1_4[['uuid', 'x0', 'y0']]

# now, we work with the Melanoma_FINAL dataset

import polars as pl


df_melanoma_final = pl.read_csv('../../melanoma/Melanoma_FINAL.csv')
df_melanoma_final.head()

from polars import col, lit

df_melanoma_final = df_melanoma_final.with_columns([
    ((col("sample_id").cast(str) + lit("_") + col("fov_id").cast(str)).alias("region_id"))
])

# create dataframes grouped by region_id for the melanoma_final dataset

# First, get a list of unique region_ids in the DataFrame
region_ids = df_melanoma_final['region_id'].unique()

# Then, create a dictionary where the keys are the region_ids and the values are the corresponding subset DataFrames
dfs = {region_id: df_melanoma_final.filter(col('region_id') == region_id) for region_id in region_ids}

# now rename the columns of the dataframes using the following dictionary. {uuid: 'CELL_ID', x0: 'X', y0: 'Y', celltype: 'CLUSTER_LABEL'}

rename_dict = {'uuid': 'CELL_ID', 'x0': 'X', 'y0': 'Y', 'celltype': 'CLUSTER_LABEL'}

# Loop over each DataFrame in the dictionary and rename the columns
for region_id, df in dfs.items():
    dfs[region_id] = df.rename(rename_dict)

# for all the dataframes (dfs), create a new dataframe for the coordinates and the uuid

## FOR COORDINATES
# Define the list of columns you want to keep
columns_to_keep = ['CELL_ID', 'X', 'Y']

# Create the new dataframes
dfs_coords = {f'df_{region_id}_coords': df.select(columns_to_keep) for region_id, df in dfs.items()}

import os

# Create the folder if it doesn't exist
folder_name = 'data_melanoma'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Write each DataFrame to a CSV file in the specified folder
for name, df in dfs_coords.items():
    df.write_csv(f'{folder_name}/{name}.csv')

## FOR CELL_TYPES
columns_to_keep_cell_type = ['CELL_ID', 'CLUSTER_LABEL']

# Create the new dataframes
dfs_cell_types = {f'df_{region_id}_cell_types': df.select(columns_to_keep_cell_type) for region_id, df in dfs.items()}

# Write each DataFrame to a CSV file in the specified folder
for name, df in dfs_cell_types.items():
    df.write_csv(f'{folder_name}/{name}.csv')
    
## FOR FEATURES
# create a column called ACQUISITION_ID in the following format: df_ + region_id

for region_id, df in dfs.items():
    dfs[region_id] = df.with_columns([
        (lit(f'df_{region_id}').alias("ACQUISITION_ID"))
    ])
    
# Define the list of columns you want to keep
columns_to_keep_features = ['ACQUISITION_ID', 'CELL_ID', 'CD45', 'CD3', 'CD4', 'CD8',
       'FOXP3', 'CD56', 'CD20', 'CD14', 'CD163', 'CD68', 'SOX10', 'S100B',
       'KI67', 'CD25', 'PD1', 'LAG3', 'TIM3', 'CD27', 'PDL1', 'B7H3', 'IDO1',
       'B2M', 'MHCI', 'MHCII', 'MRC1', 'TGM2']

# Create the new dataframes
dfs_expression = {f'df_{region_id}_expression': df.select(columns_to_keep_features) for region_id, df in dfs.items()}

# Write each DataFrame to a CSV file in the specified folder
for name, df in dfs_expression.items():
    df.write_csv(f'{folder_name}/{name}.csv')

graph_label_file = pl.read_csv('/juno/work/shah/users/ibrahih3/melanoma/MelanomaIL2_SampleAnnotations.csv')

region_ids = ['df_1_2_8',
 'df_6_3_1',
 'df_4_3_18',
 'df_3_2_23',
 'df_6_2_2',
 'df_4_2_10',
 'df_3_1_17',
 'df_5_2_1',
 'df_6_4_12',
 'df_6_2_9',
 'df_1_2_10',
 'df_5_1_3',
 'df_6_4_5',
 'df_0_2_17',
 'df_6_4_9',
 'df_6_3_7',
 'df_1_1_6',
 'df_0_2_14',
 'df_3_1_16',
 'df_0_2_13',
 'df_1_2_27',
 'df_1_2_6',
 'df_0_1_17',
 'df_3_2_33',
 'df_6_1_3',
 'df_2_4_22',
 'df_4_2_9',
 'df_1_1_8',
 'df_3_1_30',
 'df_2_1_2',
 'df_2_2_17',
 'df_2_4_33',
 'df_2_2_20',
 'df_1_3_3',
 'df_1_1_9',
 'df_4_5_3',
 'df_3_1_31',
 'df_1_2_29',
 'df_3_1_14',
 'df_2_2_2',
 'df_4_3_23',
 'df_3_2_15',
 'df_3_2_21',
 'df_2_2_3',
 'df_3_2_14',
 'df_4_5_2',
 'df_0_3_6',
 'df_0_3_21',
 'df_6_1_8',
 'df_0_2_12',
 'df_6_4_6',
 'df_3_1_22',
 'df_0_2_11',
 'df_4_4_2',
 'df_3_1_8',
 'df_3_2_31',
 'df_5_1_1',
 'df_4_2_4',
 'df_4_3_24',
 'df_4_1_2',
 'df_4_3_2',
 'df_0_1_10',
 'df_6_4_2',
 'df_4_2_3',
 'df_0_3_15',
 'df_0_1_8',
 'df_3_1_13',
 'df_4_2_7',
 'df_4_2_6',
 'df_3_1_7',
 'df_4_5_4',
 'df_3_2_20',
 'df_3_2_11',
 'df_2_1_7',
 'df_0_2_22',
 'df_2_2_11',
 'df_0_1_1',
 'df_6_1_4',
 'df_2_4_35',
 'df_4_3_14',
 'df_6_4_7',
 'df_6_1_6',
 'df_3_1_11',
 'df_4_3_1',
 'df_2_1_8',
 'df_6_3_6',
 'df_2_4_14',
 'df_4_4_3',
 'df_3_2_7',
 'df_1_2_14',
 'df_4_5_9',
 'df_4_5_10',
 'df_1_2_7',
 'df_6_3_2',
 'df_3_2_10',
 'df_2_4_6',
 'df_2_4_34',
 'df_2_4_3',
 'df_1_3_13',
 'df_2_4_2',
 'df_2_2_15',
 'df_1_2_4',
 'df_3_1_3',
 'df_0_2_29',
 'df_3_2_12',
 'df_4_5_7',
 'df_1_2_3',
 'df_3_2_3',
 'df_3_2_4',
 'df_0_2_18',
 'df_6_4_3',
 'df_0_3_10',
 'df_4_1_5',
 'df_6_3_5',
 'df_0_2_23',
 'df_6_2_7',
 'df_4_2_2',
 'df_2_2_19',
 'df_2_2_14',
 'df_3_1_35',
 'df_2_4_37',
 'df_4_4_14',
 'df_2_1_3',
 'df_1_2_2',
 'df_2_2_7',
 'df_4_4_19',
 'df_1_2_20',
 'df_6_4_4',
 'df_0_2_16',
 'df_3_1_21',
 'df_3_2_26',
 'df_1_2_15',
 'df_2_1_17',
 'df_0_1_15',
 'df_3_2_16',
 'df_3_2_5',
 'df_1_3_4',
 'df_3_2_17',
 'df_1_3_10',
 'df_0_3_9',
 'df_1_3_8',
 'df_4_1_1',
 'df_0_2_4',
 'df_3_1_4',
 'df_4_5_11',
 'df_0_2_5',
 'df_0_3_18',
 'df_3_1_28',
 'df_2_2_6',
 'df_4_3_13',
 'df_2_1_1',
 'df_2_1_11',
 'df_2_2_4',
 'df_3_1_26',
 'df_6_4_10',
 'df_1_3_5',
 'df_0_3_14',
 'df_0_1_16',
 'df_0_2_24',
 'df_1_1_7',
 'df_1_3_11',
 'df_6_1_5',
 'df_2_4_36',
 'df_2_1_15',
 'df_0_3_1',
 'df_2_4_9',
 'df_3_2_6',
 'df_1_1_5',
 'df_3_1_5',
 'df_3_2_25',
 'df_4_5_6',
 'df_6_2_6',
 'df_3_2_13',
 'df_4_2_8',
 'df_4_3_4',
 'df_0_1_6',
 'df_6_3_4',
 'df_0_1_4',
 'df_3_1_27',
 'df_3_2_30',
 'df_0_3_5',
 'df_3_1_32',
 'df_3_1_9',
 'df_0_2_30',
 'df_3_1_15',
 'df_1_3_2',
 'df_2_2_16',
 'df_0_1_25',
 'df_6_3_3',
 'df_1_2_24',
 'df_3_2_2',
 'df_6_1_2',
 'df_5_1_2',
 'df_1_2_26',
 'df_3_2_1',
 'df_5_1_4',
 'df_4_5_12',
 'df_6_1_1',
 'df_4_4_11',
 'df_4_3_20',
 'df_0_2_3',
 'df_6_4_11',
 'df_1_2_25',
 'df_0_3_12',
 'df_0_3_16',
 'df_0_3_19',
 'df_2_2_9',
 'df_3_1_33',
 'df_4_4_1',
 'df_0_3_17',
 'df_0_1_9',
 'df_2_2_18',
 'df_0_1_3',
 'df_0_3_22',
 'df_3_2_28',
 'df_3_1_10',
 'df_3_2_27',
 'df_0_2_27',
 'df_3_1_34',
 'df_1_1_1',
 'df_6_4_1',
 'df_3_1_6',
 'df_1_2_1',
 'df_2_1_6',
 'df_1_2_16',
 'df_2_4_30',
 'df_0_2_20',
 'df_0_2_19',
 'df_6_2_3',
 'df_1_2_5',
 'df_3_1_1',
 'df_0_3_3',
 'df_2_1_16',
 'df_1_1_3',
 'df_2_4_32',
 'df_0_3_8',
 'df_2_4_7',
 'df_3_1_20',
 'df_0_3_4',
 'df_2_4_4',
 'df_1_2_17',
 'df_4_1_4',
 'df_5_2_4',
 'df_6_4_13',
 'df_2_1_10',
 'df_3_2_32',
 'df_4_5_13',
 'df_4_2_11',
 'df_4_5_15',
 'df_1_1_2',
 'df_2_4_5',
 'df_3_2_29',
 'df_6_2_5',
 'df_2_1_9',
 'df_3_2_18',
 'df_3_1_18',
 'df_5_2_2',
 'df_4_5_8',
 'df_4_2_1',
 'df_2_4_23',
 'df_2_4_8',
 'df_3_1_2',
 'df_3_1_29',
 'df_0_3_20',
 'df_4_3_11',
 'df_5_2_3',
 'df_1_3_1',
 'df_1_2_9',
 'df_6_1_7',
 'df_0_3_13',
 'df_0_2_21',
 'df_2_2_10',
 'df_1_3_7',
 'df_4_3_3',
 'df_3_1_19',
 'df_0_2_28',
 'df_3_2_19',
 'df_6_2_10',
 'df_3_2_22',
 'df_3_1_12',
 'df_4_1_3',
 'df_0_3_2',
 'df_2_1_5',
 'df_1_3_6',
 'df_0_2_1',
 'df_6_2_1',
 'df_0_3_7',
 'df_1_1_4',
 'df_3_2_9',
 'df_3_2_8',
 'df_4_3_16',
 'df_4_5_5',
 'df_3_2_24',
 'df_2_2_1',
 'df_4_3_9',
 'df_3_1_23',
 'df_1_3_12',
 'df_1_3_9',
 'df_6_4_8',
 'df_0_1_7',
 'df_0_2_26',
 'df_4_5_14',
 'df_2_4_1',
 'df_4_3_17',
 'df_2_4_15',
 'df_6_2_4',
 'df_4_4_5',
 'df_4_3_15',
 'df_1_2_18',
 'df_4_2_5',
 'df_1_2_13',
 'df_4_4_17',
 'df_2_4_31',
 'df_2_1_4',
 'df_1_2_19',
 'df_4_3_19',
 'df_2_2_8',
 'df_4_1_6']    

# create a mapping between sample_id and Patient_response_abbrev from the graph_label_file dataframe
sample_id_to_patient_response_abbrev = {}
    
for i in range(len(graph_label_file)):
    sample_id_to_patient_response_abbrev[graph_label_file['sample_id'][i]] = graph_label_file['Patient_response_abbrev'][i]
    

import re

region_id_to_response = {}

for i, region_id in enumerate(region_ids):
    for sample_id, res in sample_id_to_patient_response_abbrev.items():
        pattern = f'df_{sample_id}'
        match = re.search(pattern, region_id)
        
        if match:
            region_id_to_response[region_id] = res
            
            
# create a polars dataframe from the region_id_to_response dictionary

region_ids = list(region_id_to_response.keys())
responses = list(region_id_to_response.values())
responses_num = [0 if res == 'NR' else 1 for res in responses]

# create the DataFrame
graph_label = pl.DataFrame(
    {
        "REGION_ID": region_ids,
        "RESPONSE_ABBREV": responses_num,
    }
)

# create a csv file from graph_label
folder_path = '/juno/work/shah/users/ibrahih3/codebase/space-gm/data_melanoma_label'
file_name = 'graph_label'

# write the DataFrame to a CSV file
folder_name = 'data_melanoma_label'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

graph_label.write_csv(f'{folder_path}/{file_name}.csv')

