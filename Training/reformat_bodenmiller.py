import os
import pandas as pd
import polars as pl

# Settings and global variables
pd.set_option('display.max_columns', None)
DATA_DIR = "../../data_bodenmiller/data_bodenmiller"
OUTPUT_DIR = "data_bodenmiller/basel_data"
LABEL_DIR = "data_bodenmiller_label"


def create_directory(dir_name):
    """
    Create a directory if it doesn't exist.

    Params:
    - dir_name (str): The name of the directory to be created.

    Returns:
    - None
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def read_csv_file(file_name):
    """
    Reads a CSV file from the data directory and returns it as a Polars DataFrame.

    Params:
    - file_name (str): The name of the file to be read.

    Returns:
    - pl.DataFrame: A DataFrame containing the data from the CSV file.
    """
    file_path = os.path.join(DATA_DIR, file_name)
    return pl.read_csv(file_path)

def filter_metadata(metadata, status_column, status_value, replacements=None):
    """
    Filters metadata based on a status column and applies optional string replacements.

    Params:
    - metadata (pl.DataFrame): The DataFrame to be filtered.
    - status_column (str): The name of the column to filter on.
    - status_value (str): The value to filter by in the status column.
    - replacements (dict, optional): A dictionary of values to replace in the DataFrame.

    Returns:
    - pl.DataFrame: The filtered DataFrame.
    """
    filtered_metadata = metadata.filter(pl.col(status_column) == status_value)
    if replacements:
        for old, new in replacements.items():
            filtered_metadata = filtered_metadata.with_columns(
                pl.col(old).str.replace("positive", new + "pos").str.replace("negative", new + "neg")
            )
    return filtered_metadata

def save_dataframe(df, file_name, folder=OUTPUT_DIR):
    """
    Saves a DataFrame to a CSV file in the specified folder.

    Params:
    - df (pl.DataFrame or pd.DataFrame): The DataFrame to save.
    - file_name (str): The name of the file to save the DataFrame to.
    - folder (str): The folder in which to save the file.

    Returns:
    - None
    """
    create_directory(folder)
    df.to_csv(os.path.join(folder, file_name), index=False)

def filter_data_on_column(df, column_name, filter_df, filter_column):
    """
    Filters a DataFrame based on the values in another DataFrame's column.

    Params:
    - df (pl.DataFrame): The DataFrame to be filtered.
    - column_name (str): The name of the column in df to filter on.
    - filter_df (pl.DataFrame): The DataFrame providing filter values.
    - filter_column (str): The column in filter_df to use for filtering.

    Returns:
    - pl.DataFrame: The filtered DataFrame.
    """
    return df.filter(pl.col(column_name).is_in(filter_df[filter_column]))

def process_and_save_core_data(df, core_column, value_columns, rename_dict, file_suffix):
    """
    Processes and saves core-specific data from a DataFrame.

    Params:
    - df (pl.DataFrame): The DataFrame to process.
    - core_column (str): The name of the column representing core IDs.
    - value_columns (list of str): A list of column names to include in the output.
    - rename_dict (dict): A dictionary mapping original column names to new names.
    - file_suffix (str): A suffix to append to the output file names.

    Returns:
    - None
    """
    for core in df[core_column].unique():
        core_df = df[df[core_column] == core][value_columns].rename(columns=rename_dict)
        save_dataframe(core_df, f'{core}_{file_suffix}.csv')


def main():
    # Reading data files
    antibody_panel = read_csv_file("Basel_Zuri_StainingPanel.csv")
    basel_meta = read_csv_file("Basel_PatientMetadata.csv")
    zurich_meta = read_csv_file("Zuri_PatientMetadata.csv")
    basel = read_csv_file("Basel_SC_dat.csv")  # This is the data that includes the expression values for each channel for each cell
    zurich = read_csv_file("ZurichTMA/SC_dat.csv")
    basel_coords = read_csv_file("Basel_SC_locations.csv")  # This is the data that includes the coordinates for each cell
    zurich_coords = read_csv_file("Zurich_SC_locations.csv")

    # Processing Basel metadata
    status_replacements = {"ERStatus": "ER", "PRStatus": "PR", "HER2Status": "HER2"}
    basel_meta_filtered = filter_metadata(basel_meta, "diseasestatus", "tumor", status_replacements)  # remove samples that are not tumor and simplify status columns
    basel_meta_filtered = basel_meta_filtered.with_columns(
        pl.concat_str([pl.col("ERStatus"), pl.col("PRStatus"), pl.col("HER2Status")]).alias("HR_status")
    )

    # Filter based on HR status (remove 6 patients that don't have all 3 ER, PR, and HER2 annotations)
    hr_statuses_to_exclude = ["ERposHER2neg", "HER2neg"]
    basel_meta_filtered = basel_meta_filtered.filter(
        pl.col("HR_status").is_not_in(hr_statuses_to_exclude)
    )

    # Filtering the expression and coordinates data based on the metadata (only keep samples that are in the metadata after filtering)
    basel_filtered = filter_data_on_column(basel, "core", basel_meta_filtered, "core").to_pandas()
    basel_coords_filtered = filter_data_on_column(basel_coords, "core", basel_meta_filtered, "core").to_pandas()

    # Exclude specific channels. I did this  because these channels are not relevant to the task at hand
    channels_to_exclude = ['112475Gd156Di Estroge', '10311243Ru101Di Rutheni', '10311244Ru102Di Rutheni', 
                      'I127 127II127Di', '10311239Ru96Di Rutheni', '10331253Ir191Di Iridium', 'In115 115InIn115Di', 
                      'Pb206 206PbPb206Di', '1031747Er167Di ECadhe', 'Xe131 131XeXe131Di', 'Pb204 204PbPb204Di', 
                      'Pb207 207PbPb207Di', 'Xe126 126XeXe126Di', 'Xe134 134XeXe134Di', '10311240Ru98Di Rutheni', 
                      '10311242Ru100Di Rutheni', 'phospho Histone', '10311245Ru104Di Rutheni', 'Hg202 202HgHg202Di', 
                      'ArAr80 80ArArArAr80Di', 'Pb208 208PbPb208Di', '10331254Ir193Di Iridium', '10311241Ru99Di Rutheni', 
                      'MinorAxisLength', 'EulerNumber', 'Number_Neighbors', 'Percent_Touching', 'MajorAxisLength', 
                      'Eccentricity', 'Orientation', 'Extent', 'Perimeter', 'Area', 'Solidity']
    
    basel_filtered = basel_filtered[~basel_filtered['channel'].isin(channels_to_exclude)]

    # Pivot table for wide format
    basel_filtered_wide = basel_filtered.pivot_table(index=['id', 'core'], columns='channel', values='mc_counts').reset_index()
    core_col = basel_filtered_wide.pop('core')
    basel_filtered_wide.insert(0, 'core', core_col)

    # Processing and saving core-specific data
    coords_columns = ['id', 'Location_Center_X', 'Location_Center_Y']
    coords_rename_dict = {'id': 'CELL_ID', 'Location_Center_X': 'X', 'Location_Center_Y': 'Y'}
    process_and_save_core_data(basel_coords_filtered, 'core', coords_columns, coords_rename_dict, 'coords') # save coordinates for all the cells in a single core in the form of CELL_ID, X, Yin a csv file

    expression_columns = basel_filtered_wide.columns
    expression_rename_dict = {'id': 'CELL_ID', 'core': 'ACQUISITION_ID'}
    process_and_save_core_data(basel_filtered_wide, 'core', expression_columns, expression_rename_dict, 'expression')  # save expression values for a single core in a csv file in the form of ACQUISITION_ID, CELL_ID, channel1, channel2, ...

    # Process and save labels
    basel_label = basel_meta_filtered[['core', 'HR_status']].rename(columns={'core': 'REGION_ID'})
    save_dataframe(basel_label, 'basel_label.csv', LABEL_DIR)

    # Additional processing for cell types and other specific tasks
    # Reading additional data
    basel_metaclusters = pd.read_csv("data/Bodenmiller_data/Cluster_labels/Basel_metaclusters.csv")
    basel_metacluster_annotations = pd.read_csv("data/Bodenmiller_data/Cluster_labels/Metacluster_annotations.csv")

    # Filter metaclusters based on metadata
    basel_metaclusters_filtered = basel_metaclusters[basel_metaclusters['id'].apply(lambda x: '_'.join(x.split('_')[:-1]) in basel_meta_filtered['core'].unique())]

    # Process and map metacluster annotations
    basel_metacluster_annotations['Metacluster ;Cell type;Class'] = basel_metacluster_annotations['Metacluster ;Cell type;Class'].apply(lambda x: x.split(';')[1])
    basel_metacluster_annotations.index += 1
    cluster_to_celltype_map = basel_metacluster_annotations['Metacluster ;Cell type;Class'].to_dict()

    basel_metaclusters_filtered['core'] = basel_metaclusters_filtered['id'].apply(lambda x: '_'.join(x.split('_')[:-1]))

    # Process and save cell types
    cell_type_columns = ['id', 'cluster']
    cell_type_rename_dict = {'id': 'CELL_ID', 'cluster': 'CELL_TYPE'}
    basel_metaclusters_filtered['CELL_TYPE'] = basel_metaclusters_filtered['CELL_TYPE'].map(cluster_to_celltype_map)
    process_and_save_core_data(basel_metaclusters_filtered, 'core', cell_type_columns, cell_type_rename_dict, 'cell_types')

if __name__ == "__main__":
    main()