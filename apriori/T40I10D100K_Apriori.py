# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 11:11:44 2020

@author: Luo Bingqian
"""
import pandas as pd
import time
import random
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder  # 传入模型的数据需要满足特定的格式，可以用这种方法来转换为bool值，也可以用函数转换为0、1


class T40I10D100K:

    def __init__(self):
        start = time.perf_counter()
        print("开始初始化数据集")

        with open('/Users/sgcx054/PycharmProjects/Harden/apriori/T40I10D100K.txt', 'r') as f:
            self.values=[list(map(int,''.join(line.rstrip('\n').split())))for line in f]

        #self.NumOfUser = len(self.row_groupmembership)
        self.sampleList = []
        self.SRNumOfUser = 0
        self.FIsList = []
        self.sample_FIsList = []
        #print("数据集大小为：", self.NumOfUser)

        elapsed = (time.perf_counter() - start)
        print("Time used:", elapsed)
        print("数据集已初始化完毕")

    def sample_dataset(self, SR):
        start = time.perf_counter()
        print("开始对数据集进行采样")
        print("采样率为:%.3f" % SR)

        self.SRNumOfUser = int(self.NumOfUser * SR)
        self.sampleList = random.sample(self.values, self.SRNumOfUser)

        elapsed = (time.perf_counter() - start)
        print("Time used:", elapsed)
        print("采样完毕")

    '''def deal(self,data):
        return data.dropna().tolist()'''

    def unitfiy(self):
        start = time.perf_counter()
        print("开始进一步规约数据集")

        shopping_df = pd.DataFrame(self.values)
        df_arr = shopping_df.stack().groupby(level=0).apply(list).tolist()  # 方法一
        # df_arr = shopping_df.apply(self.deal,axis=1).tolist()		        # 方法二
        te = TransactionEncoder()  # 定义模型
        df_tf = te.fit_transform(df_arr)
        # df_01 = df_tf.astype('int')			# 将 True、False 转换为 0、1 # 官方给的其它方法
        # df_name = te.inverse_transform(df_tf)		# 将编码值再次转化为原来的商品名
        self.df = pd.DataFrame(df_tf, columns=te.columns_)

        elapsed = (time.perf_counter() - start)
        print("Time used:", elapsed)
        print("数据集已进一步规约完毕")

    def unitfiy_sample_dataset(self):
        start = time.perf_counter()
        print("开始进一步规约样本数据集")

        shopping_df = pd.DataFrame(self.sampleList)
        df_arr = shopping_df.stack().groupby(level=0).apply(list).tolist()  # 方法一
        # df_arr = shopping_df.apply(self.deal,axis=1).tolist()		        # 方法二
        te = TransactionEncoder()  # 定义模型
        df_tf = te.fit_transform(df_arr)
        # df_01 = df_tf.astype('int')			# 将 True、False 转换为 0、1 # 官方给的其它方法
        # df_name = te.inverse_transform(df_tf)		# 将编码值再次转化为原来的商品名
        self.sample_df = pd.DataFrame(df_tf, columns=te.columns_)

        elapsed = (time.perf_counter() - start)
        print("Time used:", elapsed)
        print("样本数据集已进一步规约完毕")

    def getFis(self, threshold, length):

        start = time.perf_counter()
        print("开始计算频繁项集")

        print("数据集阈值为:%.3f" % threshold)
        print("数据集频繁项集长度为:%d" % length)
        frequent_itemsets = apriori(self.df, min_support=threshold,
                                    use_colnames=True)  # use_colnames=True表示使用元素名字，默认的False使用列名代表元素
        # frequent_itemsets = apriori(self.df,min_support=0.05)
        frequent_itemsets.sort_values(by='support', ascending=False, inplace=True)  # 频繁项集可以按支持度排序的
        print(frequent_itemsets[frequent_itemsets.itemsets.apply(lambda x: len(x)) >= length])  # 选择长度 >=2 的频繁项集
        #for item in frequent_itemsets.itemsets:
            #self.FIsList.append(tuple(item))
        self.FIsList =[tuple(item) for item in frequent_itemsets.itemsets]
        elapsed = (time.perf_counter() - start)
        print("Time used:", elapsed)
        print("频繁项集计算完毕")

    def getFis_sample_dataset(self, sample_threshold, sample_length):

        start = time.perf_counter()
        print("开始计算样本频繁项集")

        print("样本阈值为:%.3f" % sample_threshold)
        print("样本频繁项集长度为:%d" % sample_length)
        frequent_itemsets = apriori(self.sample_df, min_support=sample_threshold,
                                    use_colnames=True)  # use_colnames=True表示使用元素名字，默认的False使用列名代表元素
        # frequent_itemsets = apriori(self.df,min_support=0.05)
        frequent_itemsets.sort_values(by='support', ascending=False, inplace=True)  # 样本频繁项集可以按支持度排序的
        print(frequent_itemsets[frequent_itemsets.itemsets.apply(lambda x: len(x)) >= sample_length])  # 选择长度 >=2 的频繁项集
        #for item in frequent_itemsets.itemsets:
            #self.sample_FIsList.append(tuple(item))
        self.sample_FIsList=[tuple(item) for item in frequent_itemsets.itemsets]

        elapsed = (time.perf_counter() - start)
        print("Time used:", elapsed)
        print("样本频繁项集计算完毕")

    def analysis_reault(self):
        start = time.perf_counter()
        print("开始分析实验结果")

        #self.FIsList = [(1,), (2,), (1, 2), (3,), (4, 5)]
        #self.sample_FIsList = [(6,), (1, 2), (3,), (4, 5)]
        FNotsample = []
        sampleNotF = []
        '''for num in self.FIsList:
            if num not in set(self.sample_FIsList):
                FNotsample.append(num)'''
        FNotsample = [num for num in self.FIsList if num not in set(self.sample_FIsList)]
        print('频繁项个数：', len(self.FIsList))
        print('未找出的频繁项：', FNotsample)

        '''for num in self.sample_FIsList:
            if num not in set(self.FIsList):
                sampleNotF.append(num)'''
        sampleNotF = [num for num in self.sample_FIsList if num not in set(self.FIsList)]
        print('切片频繁项个数：', len(self.sample_FIsList))
        print('找错的频繁项：', sampleNotF)
        precition = (len(self.sample_FIsList) - len(sampleNotF)) / len(self.sample_FIsList)
        print('准确率为：', precition)
        recall = (len(self.FIsList) - len(FNotsample)) / len(self.FIsList)
        print('召回率为：', recall)

        elapsed = (time.perf_counter() - start)
        print("Time used:", elapsed)
        print("实验结果分析完毕")


if __name__ == '__main__':
    start = time.perf_counter()
    t40I10D100K = T40I10D100K()
    #t40I10D100K.sample_dataset(0.03)
    t40I10D100K.unitfiy()
    #t40I10D100K.unitfiy_sample_dataset()
    t40I10D100K.getFis(0.7, 1)
    #t40I10D100K.getFis_sample_dataset(0.025, 1)
    #t40I10D100K.analysis_reault()
    elapsed = (time.perf_counter() - start)
    print("Apriori Time used:", elapsed)