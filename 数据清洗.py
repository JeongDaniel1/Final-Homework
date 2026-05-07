import pandas as pd
import numpy as np
from scipy import stats
from src.utils import generate_quality_report, save_df

def load_and_clean_data(parquet_path: str):
    """M1 核心逻辑：加载→质量报告→清洗→特征工程（含注释理由）"""
    df = pd.read_parquet(parquet_path)
    print(f"原始数据规模：{df.shape[0]}行×{df.shape[1]}列")

    # 2. 生成数据质量报告（作业要求：缺失率、异常值统计）
    quality_report = generate_quality_report(df)

    # 3. 清洗策略
    # 3.1 删除关键坐标缺失
    df = df.dropna(subset=['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude'])
    # 3.2 车费合理性
    df = df[df['fare_amount'] >= 0]
    # 3.3 行程距离合理性（距离>0，避免0距离高车费的异常记录）
    df = df[df['trip_distance'] > 0]
    # 3.4 乘客数合理性
    df = df[df['passenger_count'].between(0, 6)]
    # 3.5 经纬度范围
    df = df[
        df['pickup_longitude'].between(-74.25, -73.70) &
        df['pickup_latitude'].between(40.50, 40.91)
    ]

    # 4. 特征提取
    df['pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['hour'] = df['pickup_datetime'].dt.hour  # 小时
    df['weekday'] = df['pickup_datetime'].dt.weekday  # 星期
    df['is_weekend'] = df['weekday'].isin([5,6]).astype(int)  # 是否周末
    df['is_peak'] = df['hour'].between(7,9) | df['hour'].between(17,19)  # 早晚高峰

    # 5. 自定义衍生特征
    # 5.1 行程时长
    df['trip_duration_min'] = (
        pd.to_datetime(df['tpep_dropoff_datetime']) - df['pickup_datetime']
    ).dt.total_seconds() / 60
    # 5.2 单位距离车费：反映定价合理性
    df['fare_per_mile'] = df['fare_amount'] / df['trip_distance']

    # 保存清洗后数据
    save_df(df, '../data/cleaned_taxi_data.parquet')
    return df, quality_report