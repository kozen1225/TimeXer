# Python
import argparse
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def read_swf_file(file_path):
    """读取swf文件，跳过以分号开头的注释行"""
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            if not line.startswith(';'):
                data.append(line.strip())
    return data

def drop_zero_std_columns(df):
    """删除数值标准差为0的列（不包括日期列）"""
    cols_to_drop = []
    for col in df.columns:
        if col == 'date':
            continue
        # 尝试将数据转换为数字，如果无法转换则跳过
        try:
            series = pd.to_numeric(df[col], errors='coerce')
            if series.std(skipna=True) == 0:
                cols_to_drop.append(col)
        except Exception:
            pass
    return df.drop(columns=cols_to_drop)

def calculate_submission_density(df, time_column='date', time_window=900):
    """
    使用基于事件的累积计数方法计算任务提交密度
    Args:
        df: 包含时间列的 DataFrame
        time_column: 时间列名称，默认为 'date'
        time_window: 时间窗口（秒），默认为900秒
    Returns:
        DataFrame，新增 'OT' 列
    """
    df = df.sort_values(by=time_column).reset_index(drop=True)
    df['OT'] = 0
    start_index = 0
    for end_index in range(len(df)):
        while (df[time_column].iloc[end_index] - df[time_column].iloc[start_index]).total_seconds() > time_window:
            start_index += 1
        df.loc[end_index, 'OT'] = end_index - start_index + 1
    return df

def main():
    parser = argparse.ArgumentParser(description="处理swf文件转换为csv并生成任务密度列")
    parser.add_argument('--swf', type=str, required=True, help='待处理的swf文件路径')
    parser.add_argument('--time_window', type=int, default=600, help='时间窗口大小(秒),默认为900秒')
    parser.add_argument('--basetime', type=int, help='基准时间')
    args = parser.parse_args()
    
    os.chdir('/home/gp.sc.cc.tohoku.ac.jp/dinghr1999/TimeXer/dataset/workload/')

    # 获取swf文件所在目录并切换到该目录（可选，根据需求修改）
    swf_file_path = os.path.abspath(args.swf)
    # work_dir = os.path.dirname(swf_file_path)
    # os.chdir(work_dir)

    # 读取swf文件
    data = read_swf_file(swf_file_path)

    # 转换为DataFrame，首先按空白拆分每一行，并删除第一列
    df = pd.DataFrame([x.split() for x in data])
    df = df.iloc[:, 1:]

    # 定义基准时间
    base_time = datetime.utcfromtimestamp(args.basetime)

    # 将第二列转换为数字，再转换为真实时间，后续将作为日期列
    df[1] = pd.to_numeric(df[1], errors='coerce')
    df[1] = df[1].apply(lambda x: base_time + timedelta(seconds=x))

    # 处理数值列：删除标准差为0的列，
    # 并且如果有多个列的均值和标准差完全相同，只保留第一个，删除其余的
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

    # 重命名各列，第1列为'date'，剩下的依次用'0','1',...
    df.columns = ['date'] + [str(i) for i in range(0, df.shape[1]-1)]

    # 根据日期生成任务提交密度列
    # 转换'date'列为datetime类型（可能已经是datetime，但确保一下）
    df['date'] = pd.to_datetime(df['date'])
    df = calculate_submission_density(df, time_column='date', time_window=args.time_window)

    num_cols = len(df.columns)-1

    # 保存csv，生成文件名为原swf文件名（去掉扩展名）+ _OT.csv
    base_name = os.path.splitext(os.path.basename(swf_file_path))[0]
    out_file = f"{args.time_window}_{num_cols}_{base_name}_OT.csv"
    df.to_csv(out_file, index=False)
    print(f"处理完成，结果已保存为 {out_file}")

if __name__ == '__main__':
    main()