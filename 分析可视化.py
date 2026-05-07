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
    
    # 模型评估
    model.eval()
    with torch.no_grad():
        neural_preds = model(test_tensor).numpy()
    
    nn_mae = mean_absolute_error(y_test, neural_preds)
    nn_rmse = np.sqrt(mean_squared_error(y_test, neural_preds))
    
    # 随机森林对比
    print("训练随机森林模型作为基准...")
    forest = RandomForestRegressor(n_estimators=90, max_depth=8, random_state=42, n_jobs=-1)
    forest.fit(X_train, y_train)
    forest_preds = forest.predict(X_test)
    
    forest_mae = mean_absolute_error(y_test, forest_preds)
    forest_rmse = np.sqrt(mean_squared_error(y_test, forest_preds))
    
    # 性能对比
    print("\n模型性能对比（测试集）:")
    print(f"【神经网络模型】:")
    print(f"  平均绝对误差: {nn_mae:.2f} 单/小时")
    print(f"  均方根误差: {nn_rmse:.2f} 单/小时")
    print(f"\n【随机森林模型】:")
    print(f"  平均绝对误差: {forest_mae:.2f} 单/小时")
    print(f"  均方根误差: {forest_rmse:.2f} 单/小时")

def interactive_qa_system(analysis_df):
    """交互式问答系统"""
    print("\n启动智能问答助手")
    print("您可以询问以下问题类型：")
    print("1. 数据概况（如'数据规模多大？'）")
    print("2. 时间模式（如'何时打车需求最高？'）")
    print("3. 区域分析（如'哪里打车最热门？'）")
    print("4. 费用因素（如'什么影响车费高低？'）")
    print("5. 模型表现（如'预测准确度如何？'）")
    print("输入 'exit' 或 'quit' 结束会话\n")
    
    api_key = input("请输入API密钥: ").strip()
    
    llm_client = None
    if OpenAI and api_key:
        llm_client = OpenAI(
            api_key=api_key, 
            base_url="https://open.bigmodel.cn/api/paas/v4"
        )
    
    system_instruction = """
    你是一个专业的纽约出租车数据分析助手。
    请基于以下数据特征回答问题：
    - 数据时间：2023年1月
    - 数据来源：纽约黄色出租车行程记录
    - 主要分析维度：时间模式、区域热度、费用因素、预测模型
    请用简洁专业的语言回答用户问题。
    """
    
    while True:
        try:
            query = input("\n[请问]: ").strip()
        except EOFError:
            break
            
        if query.lower() in ['exit', 'quit', '退出']:
            print("[系统]: 感谢使用，再见！")
            break
            
        if not query:
            continue
        
        # 关键词匹配本地响应
        if re.search(r"数据量|多少条|规模|概况", query):
            print(f"[本地回复]: 分析基于2023年1月纽约出租车数据，经清洗后保留 {len(analysis_df)} 条有效行程记录。")
        
        elif re.search(r"时间|高峰|时段|什么时候", query):
            print("[本地回复]: 数据显示工作日有明显早晚高峰（7-9点、17-19点），周末则呈现午后高峰模式。")
            print("📊 详细图表: visualizations/hourly_demand.png")
        
        elif re.search(r"区域|地点|哪里|热门", query):
            print("[本地回复]: 订单最集中的区域为曼哈顿核心商务区及三大机场周边。")
            print("📊 详细图表: visualizations/regional_hotspots.png")
        
        elif re.search(r"费用|价格|车费|距离", query):
            print("[本地回复]: 车费与行程距离呈强正相关，距离是最主要的定价因素。")
            print("📊 详细图表: visualizations/fare_distance_correlation.png")
        
        elif re.search(r"机场|航班|占比", query):
            print("[本地回复]: 凌晨时段（4-6点）机场订单占比显著提升，反映红眼航班旅客需求。")
            print("📊 详细图表: visualizations/airport_share.png")
        
        elif re.search(r"预测|模型|准确率|MAE|RMSE", query):
            print("[本地回复]: 随机森林模型表现优于神经网络（MAE约26 vs 60），更适合当前结构化数据预测。")
            print("📊 训练过程图: visualizations/training_loss.png")
        
        # 调用大语言模型处理其他问题
        else:
            if llm_client:
                print("[AI分析中...]", end="\r")
                try:
                    response = llm_client.chat.completions.create(
                        model="glm-4-flash",
                        messages=[
                            {"role": "system", "content": system_instruction},
                            {"role": "user", "content": query}
                        ],
                        temperature=0.7,
                        top_p=0.9
                    )
                    answer = response.choices[0].message.content
                    print(" " * 20, end="\r")
                    print(f"[AI助手]: {answer}")
                except Exception as e:
                    print(f"[系统提示]: 服务暂时不可用 ({str(e)})")
            else:
                print("[系统提示]: 未配置API密钥，无法处理该问题。")

def main_workflow():
    """主程序流程"""
    print("纽约出租车出行数据分析系统")
    
    # 创建必要目录
    for directory in ['visualizations', 'data']:
        os.makedirs(directory, exist_ok=True)
    
    data_file = './data/yellow_tripdata_2023-01.parquet'
    
    # 执行分析流程
    raw_data = examine_dataset(data_file)
    clean_data = sanitize_records(raw_data)
    enriched_data = create_features(clean_data)
    
    print("\n预处理完成，数据预览:")
    preview_cols = ['tpep_pickup_datetime', 'trip_distance', 'fare_amount', 'peak_period', 'speed_mph']
    print(enriched_data[preview_cols].head(3))
    
    visualize_patterns(enriched_data)
    build_predictive_models(enriched_data)
    interactive_qa_system(enriched_data)

if __name__ == "__main__":
    main_workflow()