import pandas as pd

def coverage_ke_3g():
    file_path = "/Users/mwangangi/Documents/Data/ITU_DH_POP_COV_3G_WIDEF.csv"
    df = pd.read_csv(file_path)

    #Filter Columns that has years incooprated
    date_columns = [c for c in df.columns if c.isdigit()]

    #Filter-Excel-file for Country Kenya
    df_kenya_data = df[df['REF_AREA_LABEL']=="Kenya"]

    #Transpose/index Rows/Column
    df_store = (df_kenya_data[date_columns].transpose().reset_index())

    df_store.columns = ["variable", "coverage_3G"]
    #Extract the year from the column name
    df_store['year'] = df_store['variable'].str.extract(r'(\d{4})').astype(int)
    print(f"Print this var (df_store['year'])\n {df_store}")
    df_store = df_store[['year','coverage_3G']].sort_values('year')

    return df_store