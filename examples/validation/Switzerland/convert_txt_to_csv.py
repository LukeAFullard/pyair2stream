import pandas as pd
import sys

def convert(txt_file, csv_file):
    df = pd.read_csv(txt_file, sep=r'\s+', header=None)
    # df columns: year, month, day, t_air, t_water, discharge
    # create 'Date' column
    df['Date'] = pd.to_datetime(dict(year=df[0], month=df[1], day=df[2]))
    df_out = pd.DataFrame()
    df_out['Date'] = df['Date']
    df_out['T_air'] = df[3]
    df_out['T_water'] = df[4]
    if df.shape[1] > 5:
        df_out['Discharge'] = df[5]
    else:
        df_out['Discharge'] = 1.0
    df_out.to_csv(csv_file, index=False)

convert('examples/validation/Switzerland/DAV_2327_cc.txt', 'examples/validation/Switzerland/DAV_2327_cc.csv')
convert('examples/validation/Switzerland/DAV_2327_cv.txt', 'examples/validation/Switzerland/DAV_2327_cv.csv')
