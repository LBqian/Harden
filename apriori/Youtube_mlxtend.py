# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 11:11:44 2020

@author: Luo Bingqian
"""
import pandas as pd
import numpy as np
import time
import random
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder	# 传入模型的数据需要满足特定的格式，可以用这种方法来转换为bool值，也可以用函数转换为0、1
class Youtube:
    DATASET_youtube_groupmembership = "./release-youtube-groupmembershipsbackup.txt"
    def __init__(self):
        start = time.clock()
        print("开始初始化数据集")
        
        self.dataset_youtube_groupmembership = np.loadtxt(self.DATASET_youtube_groupmembership)
        [self.row_groupmembership,self.col_groupmembership] = self.dataset_youtube_groupmembership.shape
        self.User = np.unique(self.dataset_youtube_groupmembership[:, 0])
        self.NumOfUser = len(self.User)
        self.dict_youtube_groupmembership={}
        self.values=[]
        self.sampleList=[]
        self.SRNumOfUser = 0
        elapsed = (time.clock() - start)
        print("Time used:",elapsed)
        print("数据集已初始化完毕")
        
    def becomedict(self):
        start = time.clock()
        print("开始生成字典并取出键值")
         
        for i in range(self.row_groupmembership):
            if self.dataset_youtube_groupmembership[i][0] in self.dict_youtube_groupmembership:
                self.dict_youtube_groupmembership[self.dataset_youtube_groupmembership[i][0]].append(self.dataset_youtube_groupmembership[i][1])
            else:
                self.dict_youtube_groupmembership[self.dataset_youtube_groupmembership[i][0]]=[self.dataset_youtube_groupmembership[i][1]]
        items=self.dict_youtube_groupmembership.items()
        for item in items:
            self.values.append(item[1])

        elapsed = (time.clock() - start)
        print("Time used:",elapsed)
        print("字典生成、键值取出")

    def sample_dataset(self,SR):
        start = time.clock()
        print("开始对数据集进行采样")

        self.SRNumOfUser = int(self.NumOfUser * SR)
        self.sampleList = random.sample(self.values,self.SRNumOfUser)

        elapsed = (time.clock() - start)
        print("Time used:", elapsed)
        print("采样完毕")

    '''def deal(self,data):
        return data.dropna().tolist()'''

    def unitfiy(self):
        start = time.clock()
        print("开始进一步规约数据集")

        shopping_df = pd.DataFrame(self.values)
        df_arr = shopping_df.stack().groupby(level=0).apply(list).tolist()	# 方法一
        #df_arr = shopping_df.apply(self.deal,axis=1).tolist()		        # 方法二
        te = TransactionEncoder()	# 定义模型
        df_tf = te.fit_transform(df_arr)
        # df_01 = df_tf.astype('int')			# 将 True、False 转换为 0、1 # 官方给的其它方法
        # df_name = te.inverse_transform(df_tf)		# 将编码值再次转化为原来的商品名
        self.df = pd.DataFrame(df_tf,columns=te.columns_)

        elapsed = (time.clock() - start)
        print("Time used:",elapsed)
        print("数据集已进一步规约完毕")

    def unitfiy_sample_dataset(self):
        start = time.clock()
        print("开始进一步规约样本数据集")

        shopping_df = pd.DataFrame(self.sampleList)
        df_arr = shopping_df.stack().groupby(level=0).apply(list).tolist()  # 方法一
        # df_arr = shopping_df.apply(self.deal,axis=1).tolist()		        # 方法二
        te = TransactionEncoder()  # 定义模型
        df_tf = te.fit_transform(df_arr)
        # df_01 = df_tf.astype('int')			# 将 True、False 转换为 0、1 # 官方给的其它方法
        # df_name = te.inverse_transform(df_tf)		# 将编码值再次转化为原来的商品名
        self.sample_df = pd.DataFrame(df_tf, columns=te.columns_)

        elapsed = (time.clock() - start)
        print("Time used:", elapsed)
        print("样本数据集已进一步规约完毕")
        
    def getFis(self,threshold,length):

        start = time.clock()
        print("开始计算频繁项集")

        print("数据集阈值为:%.3f" % threshold)
        print("数据集频繁项集长度为:%d" % length)
        frequent_itemsets = apriori(self.df,min_support=threshold,use_colnames=True)	# use_colnames=True表示使用元素名字，默认的False使用列名代表元素
        # frequent_itemsets = apriori(self.df,min_support=0.05)
        frequent_itemsets.sort_values(by='support',ascending=False,inplace=True)	# 频繁项集可以按支持度排序的
        print(frequent_itemsets[frequent_itemsets.itemsets.apply(lambda x: len(x)) >= length])  # 选择长度 >=2 的频繁项集

        elapsed = (time.clock() - start)
        print("Time used:",elapsed)
        print("频繁项集计算完毕")

    def getFis_sample_dataset(self,sample_threshold,sample_length):

        start = time.clock()
        print("开始计算样本频繁项集")

        print("样本阈值为:%.3f" %sample_threshold)
        print("样本频繁项集长度为:%d" %sample_length)
        frequent_itemsets = apriori(self.sample_df,min_support=sample_threshold,use_colnames=True)	# use_colnames=True表示使用元素名字，默认的False使用列名代表元素
        # frequent_itemsets = apriori(self.df,min_support=0.05)
        frequent_itemsets.sort_values(by='support',ascending=False,inplace=True)	# 频繁项集可以按支持度排序的
        print(frequent_itemsets[frequent_itemsets.itemsets.apply(lambda x: len(x)) >= sample_length])  # 选择长度 >=2 的频繁项集

        elapsed = (time.clock() - start)
        print("Time used:",elapsed)
        print("样本频繁项集计算完毕")

        
if __name__ == '__main__':
    start = time.clock()
    youtube = Youtube()
    youtube.becomedict()
    youtube.sample_dataset(0.5)
    #youtube.unitfiy()
    youtube.unitfiy_sample_dataset()
    #youtube.getFis()
    youtube.getFis_sample_dataset(0.018,1)
    elapsed = (time.clock() - start)
    print("Time used:",elapsed)