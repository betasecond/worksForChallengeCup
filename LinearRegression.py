# 导入需要的模块
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


# 定义计算相关系数并绘制散点图的函数
def corr_and_scatter(x,y,z):
    # This function calculates the Pearson correlation coefficient between x and y,
    # and plots a scatter plot of x and y.

    # 计算皮尔逊相关系数
    corr = x.corr(y)

    # 打印相关系数
    print("The correlation coefficient between", x.name, "and", y.name, "is:", corr)

    # 绘制散点图
    plt.scatter(z, x*y)
    plt.xlabel('edu*gdp')
    plt.ylabel(z.name)
    plt.title("The scatter plot between "  + " and " + "2012-2017"+ "GDPmultiplied by numbers of colleges")
    plt.show()

# 定义进行回归分析并绘制回归线的函数
def regression_and_plot(x, y,z):
    # This function performs a linear regression analysis on x and y,
    # and plots a regression line with the scatter plot of x and y.

    # 创建线性回归模型对象
    model = LinearRegression()

    # 拟合模型
    model.fit(z, x*y)

    # 获取截距项和斜率项
    b0 = model.intercept_
    b1 = model.coef_[0]


    #print("The regression equation is: ",,"=", ,"+",b1,"*",x.name)

    # 预测因变量
    y_pred = model.predict(x)

    # 绘制回归线和散点图
    plt.plot(x,y_pred,color="red")
    plt.scatter(x,y)
    plt.xlabel(x.name)
    plt.ylabel(y.name)
    plt.title("The regression plot between "+"Time"+" and "+"GDP multiplied by numbers of colleges")
    plt.show()


# 定义打印评估指标的函数
def print_metrics(y_true,y_pred):
   # This function prints the R-squared and mean squared error of the model.

   # 计算R方拟合度
   r2 = r2_score(y_true,y_pred)

   # 计算均方误差
   mse = mean_squared_error(y_true,y_pred)

   # 打印评估指标
   print("The R-squared of the model is:",r2)
   print("The mean squared error of the model is:",mse)
