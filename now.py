# Python
import argparse
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import holidays as hl

def read_swf_file(file_path):
    """读取swf文件，跳过以分号开头的注释行"""
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            if not line.startswith(';'):
                data.append(line.strip())
    return data

def time_switch(df, basetime):
    """
    将时间列转换为datetime类型
    """
    
    # 转换为DataFrame，首先按空白拆分每一行，并删除第一列
    df = pd.DataFrame([x.split() for x in data])
    df = df.iloc[:, 1:]
    # 定义基准时间
    base_time = datetime.utcfromtimestamp(args.basetime)

    # 将第二列转换为数字，再转换为真实时间，后续将作为日期列
    df[1] = pd.to_numeric(df[1], errors='coerce')
    df[1] = df[1].apply(lambda x: base_time + timedelta(seconds=x))

    df = drop_zero_std_columns(df)

    # 重命名各列，第1列为'date'，剩下的依次用'0','1',...
    df.columns = ['date'] + [str(i) for i in range(0, df.shape[1]-1)]
    df = df.drop(columns=['5','7']) #5列重复，7列无意义，8列已作为过滤条件
    df.columns = ['date'] + [str(i) for i in range(0, df.shape[1]-1)]

    # 将所有列转换为数字类型（除了'date'列）
    for col in df.columns:
        if col != 'date':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # 转换'date'列为datetime类型（可能已经是datetime，但确保一下）
    df['date'] = pd.to_datetime(df['date'])
    return df

def drop_zero_std_columns(df):
    """
    删除数值标准差为0的列（不包括日期列）
    并且如果有多个列的均值和标准差完全相同，只保留第一个，删除其余的
    """
    stats_dict = {}
    cols_to_drop = []
    for col in df.columns:
        if col == 'date':
            continue
        try:
            series = pd.to_numeric(df[col], errors='coerce')
            mean_val = series.mean(skipna=True)
            std_val = series.std(skipna=True)
            # 如果标准差为0，则记录删除该列
            if std_val == 0:
                cols_to_drop.append(col)
            else:
                key = (mean_val, std_val)
                if key in stats_dict:
                    cols_to_drop.append(col)
                else:
                    stats_dict[key] = col
        except Exception:
            pass
    df = df.drop(columns=cols_to_drop)
    return df


def resample_data(df, sampling_seconds) -> pd.DataFrame:
    """
    对数据进行重采样
    Args:
        df: 包含时间列的 DataFrame
        sampling_seconds: 重采样的时间间隔（秒）
    Returns:
        重采样后的 resampled_df
    """
    df['date'] = pd.to_datetime(df['date'])

    df['count'] = 1  # 添加计数列

    #此处需针对不同数据库手动修改
    aggregation_rules = {
        'wait': 'sum',              # wait
        'runtime': 'sum',           # runtime
        'cpuUsed': 'sum',           # cpu used
        'memUsed': 'sum',          # mem used
        'cpuReq': 'sum',           # cpu req
        'uid': lambda x: x.mode()[0] if not x.empty else None,           # uid
        'gid': lambda x: x.mode()[0] if not x.empty else None,           # gid
        'queue': lambda x: x.mode()[0] if not x.empty else None,         # queue
        'count': 'sum',           # count
    }

    resampled_df = df.resample(f'{sampling_seconds}s', on='date').agg(aggregation_rules)
    resampled_df.fillna(0, inplace=True)
    
    # 添加OT列，计算方式为wait/(COUNT+1)
    #resampled_df['OT'] = resampled_df['runtime'] / sampling_seconds
    #resampled_df.drop(columns=['COUNT'], inplace=True)  # 删除runtime列

    return resampled_df

def normalization(df):
    from sklearn.preprocessing import RobustScaler
    from sklearn.preprocessing import StandardScaler

    scaler_ROB = RobustScaler()
    scaler_STD = StandardScaler()

    """对数据进行归一化"""
    df['0'] = np.log1p(df['0'])
    df['0'] = scaler_ROB.fit_transform(df[['0']])

    df['1'] =scaler_ROB.fit_transform(df[['1']])

    df['2'] = np.log1p(df['2'])
    df['2'] = scaler_ROB.fit_transform(df[['2']])

    df['3'] = scaler_ROB.fit_transform(df[['3']])

    # df['OT'] = np.log1p(df['OT'])
    # df['OT'] = scaler_ROB.fit_transform(df[['OT']])

    return df

def add_holiday_column(df, country_code):
    """
    添加节假日与日期列
    Args:
        df: 包含时间列的 DataFrame
        country_code: 国家/地区代码
    Returns:
        DataFrame，新增 'Is_Holiday','Weekday_Cos','Weekday_Sin' 列
    """
    if 'date' not in df.columns:
        df = df.reset_index()

# 添加 Weekday_Sin 和 Weekday_Cos 列
    df['date'] = pd.to_datetime(df['date'])
    day_of_week = df['date'].dt.dayofweek
    weekday_sin = np.sin(2 * np.pi * day_of_week / 7)
    weekday_cos = np.cos(2 * np.pi * day_of_week / 7)

    min_year = df['date'].dt.year.min()
    max_year = df['date'].dt.year.max()
    years_range = range(min_year, max_year + 1)

    holidays = hl.country_holidays(country_code, years=years_range)
    holidays_set = set(holidays.keys())
    is_holiday = df['date'].dt.date.isin(holidays_set).astype(int)

    # 插入至'OT'之前
    insert_idx = df.columns.get_loc('OT')
    df.insert(insert_idx, 'Is_Holiday', is_holiday)
    df.insert(insert_idx, 'Weekday_Cos', weekday_cos)
    df.insert(insert_idx, 'Weekday_Sin', weekday_sin)

    return df

def main():
    parser = argparse.ArgumentParser(description="处理swf文件转换为csv并生成任务密度列")
    parser.add_argument('--swf', type=str, required=True, help='待处理的swf文件路径')
    parser.add_argument('--time_window', type=int, default=1800, help='时间窗口大小(秒),默认为1800秒')
    # parser.add_argument('--basetime', type=int, help='基准时间')
    # parser.add_argument('--country_code', type=str, default='DE', help='国家/地区代码,默认为DE')
    args = parser.parse_args()
    
    os.chdir('/home/gp.sc.cc.tohoku.ac.jp/dinghr1999/TimeXer/dataset/workload/')

    # 获取swf文件所在目录并切换到该目录（可选，根据需求修改）
    swf_file_path = os.path.abspath(args.swf)
    # work_dir = os.path.dirname(swf_file_path)
    # os.chdir(work_dir)

    # 读取swf文件
    # df = read_swf_file(swf_file_path)
    df = pd.read_csv(swf_file_path)

    for col in df.columns:
        if col != 'date':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # 转换'date'列为datetime类型（可能已经是datetime，但确保一下）
    df['date'] = pd.to_datetime(df['date'])

    # 重采样，计算COUNT
    df = resample_data(df, sampling_seconds=args.time_window)

    # 归一化
    # df = normalization(df)

    # 加入周末及节假日信息
    # df = add_holiday_column(df, country_code=args.country_code)

    base_name = os.path.splitext(os.path.basename(swf_file_path))[0]
    with open(f'ppced_{base_name}_ana.txt', 'w') as f:
        f.write(df.describe(percentiles=[.1, .25, .5, .75, .9])).to_string()
    print('describe文件已生成')
    print(df.head(5))
    print(df.describe())

    # 保存csv，生成文件名为原swf文件名（去掉扩展名）+ _OT.csv
    num_cols = len(df.columns)-1
    out_file = f"X_mean_{args.time_window}_{num_cols}_{base_name}_OT.csv"
    #out_file = f"TEMP.csv"
    df.to_csv(out_file, index=True)
    print(f"处理完成，结果已保存为 {out_file}")

if __name__ == '__main__':
    main()