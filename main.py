import LinearRegression as Lin
#导入functions.py文件


# =数据=存储在df中
import pandas as pd # 导入pandas模块
df = pd.read_csv("data.csv") # 读取data.csv文件并赋值给df变量
# 调用上面定义的三个函数
Lin.corr_and_scatter(df["edu"], df["gdp"],df["time"])  # 计算相关系数并绘制散点图
Lin.regression_and_plot(df[["edu"]], df["gdp"],df["time"])  # 进行回归分析并绘制回归线
Lin.print_metrics(df["gdp"], Lin.model.predict(df[["edu"]]))  # 打印评估指标

