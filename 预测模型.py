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

class TripDemandNet(nn.Module):
    """行程需求预测神经网络"""
    def __init__(self, input_features):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_features, 48),
            nn.ReLU(),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

def build_predictive_models(feature_df):
    """训练和评估预测模型"""
    print("\n构建需求预测模型...")
    
    # 准备建模数据
    feature_df['travel_date'] = feature_df['tpep_pickup_datetime'].dt.date
    aggregated = feature_df.groupby([
        'travel_date', 'hour_of_day', 'PULocationID', 
        'day_of_week', 'peak_period', 'weekend_flag'
    ]).size().reset_index(name='order_count')
    
    # 选择最活跃的区域
    busy_locations = aggregated.groupby('PULocationID')['order_count'].sum().nlargest(20).index
    modeling_data = aggregated[aggregated['PULocationID'].isin(busy_locations)]
    
    # 准备特征和标签
    predictors = ['PULocationID', 'hour_of_day', 'day_of_week', 'weekend_flag', 'peak_period']
    X = modeling_data[predictors].values
    y = modeling_data['order_count'].values
    
    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 标准化处理
    normalizer = StandardScaler()
    X_train_norm = normalizer.fit_transform(X_train)
    X_test_norm = normalizer.transform(X_test)
    
    # 转换为PyTorch张量
    train_tensor = torch.tensor(X_train_norm, dtype=torch.float32)
    train_labels = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    test_tensor = torch.tensor(X_test_norm, dtype=torch.float32)
    test_labels = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    
    # 初始化神经网络
    model = TripDemandNet(input_dim=X_train.shape[1])
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.008)
    
    # 训练循环
    training_history = []
    for epoch in range(180):
        predictions = model(train_tensor)
        loss = loss_fn(predictions, train_labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        training_history.append(loss.item())
        if (epoch + 1) % 45 == 0:
            print(f"     训练轮次 [{epoch+1}/180], 损失值: {loss.item():.4f}")
    
    # 保存训练过程图
    plt.figure(figsize=(8, 5))
    plt.plot(training_history, linewidth=2)
    plt.title('神经网络训练损失变化')
    plt.xlabel('训练轮次')
    plt.ylabel('均方误差损失')
    plt.grid(alpha=0.3)
    plt.savefig('visualizations/training_loss.png', dpi=300)
    plt.close()
    print("  -> 训练过程图保存至 visualizations/training_loss.png")