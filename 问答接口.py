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