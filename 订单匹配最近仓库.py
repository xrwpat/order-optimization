# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import datetime
from datetime import datetime



#数据读取
#数据读取
order=pd.read_excel('零售小票.xlsx')
store=pd.read_excel('仓库档案.xlsx')
location=pd.read_csv('经纬度省级汇总.csv')
stock=pd.read_excel('库存数据.xlsx')



#经纬度数据处理
for i in range(0,len(location.province)):
    location.province[i]=location.province[i][0:2]



#仓库数据处理
store['latitude']=0
store['longitude']=0
for i in range(0,len(store.仓库名称)):
    store.仓库名称[i]=store.仓库名称[i][0:2]
    store['latitude'][i]=location[location['province']==store['仓库名称'][i]]['latitude']
    store['longitude'][i]=location[location['province']==store['仓库名称'][i]]['longitude']
store.head()


#订单数据处理
order['latitude']=0
order['longitude']=0
for i in range(0,1000):
    order.收货地址[i]=order.收货地址[i][0:2]
    order['latitude'][i]=location[location['province']==order['收货地址'][i]]['latitude']
    order['longitude'][i]=location[location['province']==order['收货地址'][i]]['longitude']


#筛选订单中只有一个商品的订单：
order=order[0:10000]#先测试前10000个订单
order_sum=pd.DataFrame(order[0:10000].groupby('单据编号')['数量'].sum()).reset_index()
order_sum.head()
order_sum_new=order_sum[order_sum['数量']==1]
order_sum_new.head()


order_223_for_test=pd.DataFrame()
for i in order_sum_new['单据编号']:
    order_223_for_test=pd.concat([order[order['单据编号']==i],order_223_for_test])
    print(order_223_for_test)


#保存223个测试订单。
# order_223_for_test.to_csv('/Users/xie/Downloads/Qishon/订单路由问题/1000个测试订单.csv')
#程序测试从这里开始。



start=datetime.now()


#定义一个订单与各仓库之间的最近距离的函数。输入目标地点经纬度，返回距离。
def sort_by_value(d): 
    items=d.items() 
    backitems=[[v[1],v[0]] for v in items] 
    backitems.sort() 
    return [ backitems[i][1] for i in range(0,len(backitems))]

def distance_calcutlate(order_ID):
    distance_dict={}
    latitude=float(order_223_for_test[order_223_for_test['单据编号']==order_ID]['latitude'])
    longitude=float(order_223_for_test[order_223_for_test['单据编号']==order_ID]['longitude'])
    for i in location['province']:
        latitude_1=location[location['province']==i]['latitude']
        longitude_1=location[location['province']==i]['longitude']
        distance=int((latitude-latitude_1)*(latitude-latitude_1)+(longitude-longitude_1)*(longitude-longitude_1))
        distance_dict[i]=distance
    return(distance_dict)


def sort_by_value(d): 
    items=d.items() 
    backitems=[[v[1],v[0]] for v in items] 
    backitems.sort() 
    return [ backitems[0][1]]




for order_ID in order_sum_new['单据编号']:
    a=order_223_for_test[order_223_for_test['单据编号']==order_ID].index
    商品代码=order_223_for_test['商品代码'][a.values[0]]
    颜色代码=order_223_for_test['颜色代码'][a.values[0]]
    尺码代码=order_223_for_test['尺码代码'][a.values[0]]
    current_stock=stock[(stock['商品代码']==商品代码)&(stock['颜色代码']==颜色代码)&(stock['尺码代码']==尺码代码)]
    for i in sorted(distance_calcutlate(order_ID).items(), key=lambda item:item[1]):
        warehouse=i[0]
        if warehouse in list(current_stock['仓库名称']):
            #修改库存
            condition=(stock['仓库名称']==warehouse) & (stock['商品代码'] == 商品代码) & (stock['颜色代码'] == 颜色代码) & (stock['尺码代码']==尺码代码)
            stock['数量'][stock[condition==True].index[0]]=stock['数量'][stock[condition==True].index[0]]-1
            if stock['数量'][stock[condition==True].index[0]]>0:
                print('最新库存：',stock['数量'][stock[condition==True].index[0]])
                print(order_ID,'最近的仓库是：',warehouse)
            else:
                print(order_ID,'库存不足')
            break





stop=datetime.now()
print('time:',stop-start)

