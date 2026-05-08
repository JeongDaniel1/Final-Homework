import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import re
from datetime import datetime
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# 配置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def examine_dataset(file_location):
    """加载并检查数据集质量"""
    dataset = pd.read_parquet(file_location)
    print(f"数据集初始规模: {len(dataset)} 条记录")
    
    # 计算各列空值比例
    null_percentages = (dataset.isnull().sum() / len(dataset) * 100).round(2)
    print("\n存在空值的字段及其比例 (%):")
    print(null_percentages[null_percentages > 0])
    
    # 关键数值字段的统计摘要
    print("\n关键数值字段分布统计:")
    numeric_fields = ['trip_distance', 'fare_amount', 'passenger_count']
    print(dataset[numeric_fields].describe())
    return dataset

def sanitize_records(raw_df):
    """
    数据清洗流程：
    1. 移除无效交易：车费≤0、距离≤0或≥1000英里
    2. 过滤异常乘客数：0人或超过9人
    3. 纠正时间逻辑：确保下车时间晚于上车时间
    4. 剔除极端时长：行程超过10小时的记录
    """
    original_count = len(raw_df)
    
    # 基础条件筛选
    mask = (
        (raw_df['fare_amount'] > 0) &
        (raw_df['trip_distance'] > 0) & 
        (raw_df['trip_distance'] < 1000) &
        (raw_df['passenger_count'] > 0) & 
        (raw_df['passenger_count'] <= 9) &
        (raw_df['tpep_dropoff_datetime'] > raw_df['tpep_pickup_datetime'])
    )
    cleaned_df = raw_df.loc[mask].copy()
    
    # 计算行程时长（小时）
    trip_hours = (cleaned_df['tpep_dropoff_datetime'] - 
                  cleaned_df['tpep_pickup_datetime']).dt.total_seconds() / 3600
    cleaned_df = cleaned_df[trip_hours <= 10]
    
    removed = original_count - len(cleaned_df)
    print(f"数据清洗完成：移除了 {removed} 条异常记录，保留 {len(cleaned_df)} 条有效数据")
    return cleaned_df

def create_features(source_df):
    """从原始数据中提取有用特征"""
    pickup_time = source_df['tpep_pickup_datetime']
    
    # 时间相关特征
    source_df['hour_of_day'] = pickup_time.dt.hour
    source_df['day_of_week'] = pickup_time.dt.weekday  # 周一=0, 周日=6
    source_df['weekend_flag'] = (source_df['day_of_week'] >= 5).astype(int)
    
    # 高峰期标识（工作日早晚高峰）
    peak_condition = (
        (source_df['weekend_flag'] == 0) &
        ((source_df['hour_of_day'].between(7, 9)) | 
         (source_df['hour_of_day'].between(17, 19)))
    )
    source_df['peak_period'] = peak_condition.astype(int)
    
    # 行程特征
    source_df['duration_minutes'] = (
        source_df['tpep_dropoff_datetime'] - pickup_time
    ).dt.total_seconds() / 60
    
    # 计算平均速度（英里/小时）
    source_df['speed_mph'] = source_df['trip_distance'] / (source_df['duration_minutes'] / 60)
    
    # 移除速度异常值
    source_df = source_df[source_df['speed_mph'] <= 120].copy()
    return source_df
