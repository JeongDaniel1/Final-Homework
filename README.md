# Final-Homework
城市出租车出行数据分析与智能问答系统
#1.AI 交互日志：
按模块记录关键 Prompt、AI 输出摘要及你的采用/修改/拒绝决策；重点记录 AI 犯错的案例（错在哪、如何发现、如何修正）；
在M2中，我的prompt:我想看不同行政区的单价差异，用什么图？给我一段seaborn代码，要分工作日和周末。
AI摘要：建议用Barplot，给出了分组聚合的代码，比如按Borough分组算平均车费。
但是运行中一直出现问题，于是我一步步检查代码（真的很艰辛），发现AI的代码没处理异常值，比如有天价车费，我加了把车费限制在50美元以内，不然图Y轴太高，看不出分布。

#2.三阶段对比：
选取一个典型功能，展示 Native 版（独立写）、Prompt 版（AI辅助）、Vibe 版（对话驱动）三个版本的代码差异，分析各自的效率与理解深度
我从中找了一个最典型的，发现AI想多比我全面多了。
Native:
python# 直接用原始特征训练，没做任何预处理
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
Prompt：
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
Vibe:
from sklearn.preprocessing import StandardScaler
import joblib
#分离数值特征和类别特征（关键调整）
numeric_features = ['hour_of_day', 'day_of_week']  # 数值型：小时、星期
categorical_features = ['PULocationID', 'weekend_flag', 'peak_period']  # 类别型：区域ID、周末标志、高峰标# 只对数值特征标准化（保留类别特征的原始语义）
X_train_numeric = X_train[:, [list(X_train.columns).index(f) for f in numeric_features]]
X_train_cat = X_train[:, [list(X_train.columns).index(f) for f in categorical_features]]

scaler = StandardScaler()
X_train_numeric_scaled = scaler.fit_transform(X_train_numeric)
X_train_scaled = np.hstack([X_train_cat, X_train_numeric_scaled])  # 合并特征# 保存scaler用于后续预测（避免数据泄露）
joblib.dump(scaler, 'models/scaler.pkl')
分析效率：让我想到这门课倡导的人机协作，这大概就是较好的状态，不过度使用AI，我负责考虑新的想法，AI负责来做。
Native：我相比于AI考虑的太少了，容易出现bug。
Prompt：展现了AI相对于我这种初学者的优势：短时间内生成严谨的代码，令我五体投地。
Native：
#3.反思：
完成本次作业后，你对 AI 工具的能力边界有什么新的认识？（不少于 150 字）
现在的AI在编写代码领域的效率、速度与准确率远高于人类，因此，人们可以通过喂给AI关键词与关键结构，让AI短时间写出几百行代码，解放生产力。
此外，AI也能辅助人类进行代码检查，检查潜在的漏洞。现在中国的AI模型千问等大模型可以直接插在VScode中，方便人们在编码时随时提问。
AI也有能力的边界，他并不能做到100%让代码运行成功，有些逻辑仍需要人类的智慧。
此次大作业也让我学习了正确使用AI的方法。我在向AI根据作业要求提问时，AI习惯于用冷门的数据库来玩成1内容，因此需要我不断追问用所学的语言去完成作业。同时，AI也是我的好老师，替我检查error，同时提供更好的逻辑建议，也让我体会到人工智能在编写基础代码领域真的领先我（初学者）许多。
