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


def visualize_patterns(processed_df):
    """创建数据分析可视化图表"""
    print("\n生成分析图表中...")
    output_folder = 'visualizations/'
    os.makedirs(output_folder, exist_ok=True)
    
    # 图1：每日时段订单分布
    plt.figure(figsize=(10, 6))
    processed_df['date'] = processed_df['tpep_pickup_datetime'].dt.date
    
    # 计算每日各时段平均订单量
    daily_pattern = processed_df.groupby(['date', 'hour_of_day', 'weekend_flag']).size().reset_index(name='orders')
    hourly_avg = daily_pattern.groupby(['weekend_flag', 'hour_of_day'])['orders'].mean().reset_index()
    
    sns.lineplot(data=hourly_avg, x='hour_of_day', y='orders', 
                 hue='weekend_flag', marker='o', palette=['steelblue', 'coral'])
    plt.title('不同时段平均订单量对比')
    plt.xlabel('小时 (0-23)')
    plt.ylabel('平均订单数')
    plt.xticks(range(0, 24, 2))
    plt.legend(title='日期类型', labels=['工作日', '周末'])
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_folder, 'hourly_demand.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 图2：热门区域订单分布
    plt.figure(figsize=(12, 6))
    top_regions = processed_df['PULocationID'].value_counts().head(10).index
    region_data = processed_df[processed_df['PULocationID'].isin(top_regions)]
    
    region_summary = region_data.groupby(['PULocationID', 'peak_period']).size().unstack(fill_value=0)
    region_summary['total_orders'] = region_summary.sum(axis=1)
    region_summary = region_summary.sort_values('total_orders', ascending=False).drop(columns='total_orders')
    
    region_summary.plot(kind='bar', stacked=True, color=['lightblue', 'salmon'])
    plt.title('十大热门区域订单时段分布')
    plt.xlabel('区域编号')
    plt.ylabel('订单总量')
    plt.legend(['平峰期', '高峰期'])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'regional_hotspots.png'), dpi=300)
    plt.close()
    
    # 图3：费用与距离关系
    plt.figure(figsize=(10, 6))
    sample_subset = processed_df.sample(min(10000, len(processed_df)), random_state=42)
    
    scatter = sns.scatterplot(data=sample_subset, x='trip_distance', y='fare_amount', 
                             hue='passenger_count', palette='coolwarm', alpha=0.6, s=15)
    plt.title('行程距离与费用相关性分析')
    plt.xlabel('行驶距离 (英里)')
    plt.ylabel('车费金额 ($)')
    
    # 限制坐标轴范围以突出主要分布
    x_limit = sample_subset['trip_distance'].quantile(0.99)
    y_limit = sample_subset['fare_amount'].quantile(0.99)
    plt.xlim(0, x_limit)
    plt.ylim(0, y_limit)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_folder, 'fare_distance_correlation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 图4：机场订单时段占比
    plt.figure(figsize=(10, 6))
    airport_codes = [1, 132, 138]  # EWR, JFK, LGA
    processed_df['airport_pickup'] = processed_df['PULocationID'].isin(airport_codes)
    
    hourly_airport = processed_df.groupby('hour_of_day').agg(
        total_orders=('PULocationID', 'count'),
        airport_orders=('airport_pickup', 'sum')
    ).reset_index()
    hourly_airport['airport_percentage'] = (hourly_airport['airport_orders'] / hourly_airport['total_orders'] * 100).round(1)
    
    # 组合柱状图和折线图
    ax = sns.barplot(data=hourly_airport, x='hour_of_day', y='airport_percentage', color='lightcyan')
    ax2 = ax.twinx()
    sns.lineplot(data=hourly_airport, x='hour_of_day', y='airport_percentage', 
                ax=ax2, color='navy', marker='D', linewidth=2)
    
    plt.title('24小时机场订单占比变化')
    ax.set_xlabel('小时 (0-23)')
    ax.set_ylabel('机场订单占比 (%)')
    ax.set_xticks(range(0, 24, 2))
    
    # 添加数值标签
    for idx, value in enumerate(hourly_airport['airport_percentage']):
        ax.text(idx, value + 0.3, f"{value}%", ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'airport_share.png'), dpi=300)
    plt.close()

