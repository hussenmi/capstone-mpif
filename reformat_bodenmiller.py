import pandas as pd
import polars as pl

pd.set_option('display.max_columns', None)

antibody_panel = pl.read_csv("data/Bodenmiller_data/Data_publication/Basel_Zuri_StainingPanel.csv")

# metadata for basel and zurich
basel_meta = pl.read_csv("data/Bodenmiller_data/Data_publication/BaselTMA/Basel_PatientMetadata.csv")
zurich_meta = pl.read_csv("data/Bodenmiller_data/Data_publication/ZurichTMA/Zuri_PatientMetadata.csv")


# single-cell data for basel and zurich
basel = pl.read_csv("data/Bodenmiller_data/Data_publication/BaselTMA/SC_dat.csv")
zurich = pl.read_csv("data/Bodenmiller_data/Data_publication/ZurichTMA/SC_dat.csv")

basel_coords = pl.read_csv("data/Bodenmiller_data/Basel_SC_locations.csv")
zurich_coords = pl.read_csv("data/Bodenmiller_data/Zurich_SC_locations.csv")

### cleaning up basel
# remove normal samples
basel_meta_filtered = basel_meta.filter(pl.col("diseasestatus") == "tumor")
# create HR status column (concatenating ER/PR/HER2)
basel_meta_filtered = basel_meta_filtered.with_columns(pl.col("ERStatus").str.replace("positive", "ERpos"))
basel_meta_filtered = basel_meta_filtered.with_columns(pl.col("ERStatus").str.replace("negative", "ERneg"))
basel_meta_filtered = basel_meta_filtered.with_columns(pl.col("PRStatus").str.replace("positive", "PRpos"))
basel_meta_filtered = basel_meta_filtered.with_columns(pl.col("PRStatus").str.replace("negative", "PRneg"))
basel_meta_filtered = basel_meta_filtered.with_columns(pl.col("HER2Status").str.replace("positive", "HER2pos"))
basel_meta_filtered = basel_meta_filtered.with_columns(pl.col("HER2Status").str.replace("negative", "HER2neg"))
basel_meta_filtered = basel_meta_filtered.with_columns(pl.concat_str([pl.col("ERStatus"),pl.col("PRStatus"),pl.col("HER2Status"),])
                                 .alias("HR_status"),)
# remove 6 patients that don't have all 3 ER, PR, and HER2 annotations
basel_meta_filtered = basel_meta_filtered.filter(pl.col("HR_status") != "ERposHER2neg")
basel_meta_filtered = basel_meta_filtered.filter(pl.col("HR_status") != "HER2neg")

# target categories
list(basel_meta_filtered["HR_status"].unique())
# remove samples in basel single-cell data that are not in basel_meta_filtered prior to running GNN
### TO DO
# from basel, only select rows that are in basel_meta_filtered (in the 'core' column)
from polars import col, lit
basel_filtered = basel.filter(pl.col("core").is_in(basel_meta_filtered["core"]))
basel_coords_filtered = basel_coords.filter(pl.col("core").is_in(basel_meta_filtered["core"]))

# convert them into pandas dataframes

basel_meta_filtered = basel_meta_filtered.to_pandas()
basel_filtered = basel_filtered.to_pandas()
basel_coords_filtered = basel_coords_filtered.to_pandas()

channels_to_exclude = ['112475Gd156Di Estroge', '10311243Ru101Di Rutheni', '10311244Ru102Di Rutheni', 
                      'I127 127II127Di', '10311239Ru96Di Rutheni', '10331253Ir191Di Iridium', 'In115 115InIn115Di', 
                      'Pb206 206PbPb206Di', '1031747Er167Di ECadhe', 'Xe131 131XeXe131Di', 'Pb204 204PbPb204Di', 
                      'Pb207 207PbPb207Di', 'Xe126 126XeXe126Di', 'Xe134 134XeXe134Di', '10311240Ru98Di Rutheni', 
                      '10311242Ru100Di Rutheni', 'phospho Histone', '10311245Ru104Di Rutheni', 'Hg202 202HgHg202Di', 
                      'ArAr80 80ArArArAr80Di', 'Pb208 208PbPb208Di', '10331254Ir193Di Iridium', '10311241Ru99Di Rutheni', 
                      'MinorAxisLength', 'EulerNumber', 'Number_Neighbors', 'Percent_Touching', 'MajorAxisLength', 
                      'Eccentricity', 'Orientation', 'Extent', 'Perimeter', 'Area', 'Solidity']

# from basel_filtered, remove rows that have their 'channel' column in channels_to_exclude
basel_filtered = basel_filtered[~basel_filtered['channel'].isin(channels_to_exclude)]

# create a dictionary where the keys are the unique cores and the values are the dataframes containing the individual cell locations
basel_coords_filtered_dict = {}

for i in basel_coords_filtered['core'].unique():
    basel_coords_filtered_dict[i] = basel_coords_filtered[basel_coords_filtered['core'] == i]
    basel_coords_filtered_dict[i] = basel_coords_filtered_dict[i][['id', 'Location_Center_X', 'Location_Center_Y']]
    basel_coords_filtered_dict[i] = basel_coords_filtered_dict[i].rename(columns={'id': 'CELL_ID', 'Location_Center_X': 'X', 'Location_Center_Y': 'Y'})

# create csv files for the coords groped by the core they belong in
import os

folder_name = 'data_bodenmiller/basel_data'

if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    
# for each dataframe in the dictionary, save it as a csv in the specified folder

for core_name in basel_coords_filtered_dict:
    basel_coords_filtered_dict[core_name].to_csv(f'{folder_name}/{core_name}_coords.csv', index=False)

# for the basel_filtered dataframe, make it a wide format to have the features as columns
basel_filtered_wide = basel_filtered.pivot_table(index=['id', 'core'], columns='channel', values='mc_counts').reset_index()

core_col = basel_filtered_wide.pop('core')
basel_filtered_wide.insert(0, 'core', core_col)

# for the basel_filtered_wide dataframe, create a dictionary of dataframes for each core, where the key is the core name and the value is the dataframe containing the core name, id, and the features as columns

basel_filtered_wide_dict = {}

for i in basel_filtered_wide['core'].unique():
    basel_filtered_wide_dict[i] = basel_filtered_wide[basel_filtered_wide['core'] == i]
    basel_filtered_wide_dict[i] = basel_filtered_wide_dict[i].rename(columns={'id': 'CELL_ID', 'core': 'ACQUISITION_ID'})
    

# for each dataframe in the dictionary, save it as a csv in the specified folder
for core_name in basel_filtered_wide_dict:
    basel_filtered_wide_dict[core_name].to_csv(f'{folder_name}/{core_name}_expression.csv', index=False)


# now, we create the dataframe containing the label for each core

basel_label = basel_meta_filtered[['core', 'HR_status']].rename(columns={'core': 'REGION_ID'})
basel_label_new = basel_meta_filtered[['core', 'ERStatus', 'PRStatus', 'HER2Status', 'HR_status']].rename(columns={'core': 'REGION_ID'})

# in basel_label_new, for the columns ERStatus, PRStatus, and HER2Status, replace the values with pos and neg with 1 and 0, respectively
basel_label_new.loc[basel_label_new['ERStatus'] == 'ERpos', 'ERStatus'] = 1
basel_label_new.loc[basel_label_new['ERStatus'] == 'ERneg', 'ERStatus'] = 0

basel_label_new.loc[basel_label_new['PRStatus'] == 'PRpos', 'PRStatus'] = 1
basel_label_new.loc[basel_label_new['PRStatus'] == 'PRneg', 'PRStatus'] = 0

basel_label_new.loc[basel_label_new['HER2Status'] == 'HER2pos', 'HER2Status'] = 1
basel_label_new.loc[basel_label_new['HER2Status'] == 'HER2neg', 'HER2Status'] = 0

# save the label dataframe as a csv in the current working directory

folder_name_label = 'data_bodenmiller_label'

if not os.path.exists(folder_name_label):
    os.makedirs(folder_name_label)
    
basel_label_new.to_csv(f'{folder_name_label}/basel_label.csv', index=False)

# cell types
basel_metaclusters = pd.read_csv("data/Bodenmiller_data/Cluster_labels/Basel_metaclusters.csv")

basel_metacluster_annotations = pd.read_csv("data/Bodenmiller_data/Cluster_labels/Metacluster_annotations.csv")

basel_metaclusters_filtered = basel_metaclusters[basel_metaclusters['id'].apply(lambda x: '_'.join(x.split('_')[:-1]) in basel_meta_filtered['core'].unique())]
basel_metacluster_annotations['Metacluster ;Cell type;Class'] = basel_metacluster_annotations['Metacluster ;Cell type;Class'].apply(lambda x: x.split(';')[1])

basel_metacluster_annotations.index = basel_metacluster_annotations.index + 1
cluster_to_celltype_map = basel_metacluster_annotations['Metacluster ;Cell type;Class'].to_dict()

basel_metaclusters_filtered['core'] = basel_metaclusters_filtered['id'].apply(lambda x: '_'.join(x.split('_')[:-1]))

# for basel_metaclusters_filtered, create a dictionary of dataframes for each core, where the key is the core name and the value is the dataframe containing the id, and the cluster as columns

basel_metaclusters_filtered_dict = {}

for i in basel_metaclusters_filtered['core'].unique():
    basel_metaclusters_filtered_dict[i] = basel_metaclusters_filtered[basel_metaclusters_filtered['core'] == i]
    basel_metaclusters_filtered_dict[i] = basel_metaclusters_filtered_dict[i][['id', 'cluster']]
    basel_metaclusters_filtered_dict[i] = basel_metaclusters_filtered_dict[i].rename(columns={'id': 'CELL_ID', 'cluster': 'CELL_TYPE'})
    
    basel_metaclusters_filtered_dict[i]['CELL_TYPE'] = basel_metaclusters_filtered_dict[i]['CELL_TYPE'].map(cluster_to_celltype_map)

# for each dataframe in the dictionary, save it as a csv in the specified folder

for core_name in basel_metaclusters_filtered_dict:
    basel_metaclusters_filtered_dict[core_name].to_csv(f'{folder_name}/{core_name}_cell_types.csv', index=False)