import pandas as pd
import numpy as np
import datetime
from datetime import datetime

#数据读取
print('data loading……')
order=pd.read_excel('零售小票.xlsx')
store=pd.read_excel('仓库档案.xlsx')
stock=pd.read_excel('库存数据.xlsx')
location=pd.read_csv('经纬度省级汇总.csv')
print('loading compeleted')

print(location.head())


#经纬度数据处理
for i in range(0,len(location.province)):
    location.province[i]=location.province[i][0:2]

#筛选订单中只有一个商品的订单：
order=order[0:10000]#先测试前10000个订单
order_sum=pd.DataFrame(order.groupby('单据编号')['数量'].sum()).reset_index()
order_sum.head()
order_sum_new=order_sum[order_sum['数量']==1]
order_sum_new.head()
len(order_sum_new)
order_2490_for_test=pd.DataFrame()
for i in order_sum_new['单据编号']:
    order_2490_for_test=pd.concat([order[order['单据编号']==i],order_2490_for_test])

#______________________________________________
def distance_calcutlate(order_ID):
    distance_dict={}
    latitude=float(order_2490_for_test[order_2490_for_test['单据编号']==order_ID]['latitude'])
    longitude=float(order_2490_for_test[order_2490_for_test['单据编号']==order_ID]['longitude'])
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

def order_allocation(index,order_table,stock_table):
    for order_ID in index:
        index=order_table[order_table['单据编号']==order_ID].index[0]#取出索引
        SKU_ID=order_table['商品代码'][index]
        try:
            color_ID=int(order_table['颜色代码'][index])
        except:
            color_ID=order_table['颜色代码'][index][-2:-1]
        size_ID=order_table['尺码代码'][index]
        condition=(stock_table['商品代码']==SKU_ID) & (stock_table['颜色代码']==int(color_ID))&(stock_table['尺码代码']==int(size_ID))
        current_stock=stock_table[(stock_table['商品代码']==SKU_ID) & (stock_table['颜色代码']==int(color_ID))&(stock_table['尺码代码']==int(size_ID))]
        for i in sorted(distance_calcutlate(order_ID).items(), key=lambda item:item[1]):
            warehouse=i[0][0:2]
            if warehouse in list(current_stock['仓库名称']):
                #修改库存
                stock_table['数量'][stock_table[condition==True].index[0]]=stock_table['数量'][stock_table[condition==True].index[0]]-1
#                 if stock_table['数量'][stock_table[condition==True].index[0]]>0:
#                     print('最新库存：',stock_table['数量'][stock_table[condition==True].index[0]])
#                     print(order_ID,'最近的仓库是：',warehouse)
#                 else:
#                     print(order_ID,'库存不足')
                break

#————————————————————————————————————————————————————
#三进程用时测试
# 开始计时
start = datetime.now()

# 订单重复率统计，初步分批
订单重复率统计 = pd.DataFrame(order_2490_for_test.groupby('商品代码、颜色、尺码')['数量'].sum()).reset_index()
batch1 = pd.DataFrame()
batch2 = pd.DataFrame()
for i in range(0, len(订单重复率统计)):
    sku_color_size = 订单重复率统计[订单重复率统计.columns[0]][i]
    if 订单重复率统计[订单重复率统计.columns[1]][i] > 1:
        batch1 = pd.concat(
            [batch1, order_2490_for_test.loc[order_2490_for_test['商品代码、颜色、尺码'] == sku_color_size]])  # batch1是重复的订单
    else:
        batch2 = pd.concat([batch2, order_2490_for_test.loc[
            order_2490_for_test['商品代码、颜色、尺码'] == sku_color_size]])  # batch2是非重复的订单，可以将batch2划分为多个同时处理

# 将不重复的订单再次分批
batch2_1 = batch2[0:500]
batch2_2 = batch2[501:1000]
batch2_3 = batch2[1001:]

# 运行batch1
start1 = datetime.now()
order_allocation(batch1['单据编号'], batch1, stock)
stop1 = datetime.now()
print('batch1用时：', stop1 - start1)

# 运行batch2
start2 = datetime.now()
from multiprocessing import Process, Queue
import os, time, random

if __name__ == '__main__':
    # 父进程创建Queue，并传给各个子进程：
    q = Queue()
    p1 = Process(target=order_allocation, args=(batch2_1['单据编号'], batch2_1, stock))
    p2 = Process(target=order_allocation, args=(batch2_2['单据编号'], batch2_2, stock))
    p3 = Process(target=order_allocation, args=(batch2_3['单据编号'], batch2_3, stock))
    # 启动子进程p1
    p1.start()
    # 启动子进程p2
    p2.start()
    # 启动子进程p3
    p3.start()
    # 等待p1,p2,p3结束:
    p1.join()
    p2.join()
    p3.join()
stop2 = datetime.now()
print('batch2用时：', stop2 - start2)

stop = datetime.now()
print('开三线程总用时：', stop - start)



#——————————————————————————————————————————————————
#单进程用时测试
start3=datetime.now()
order_allocation(order_2490_for_test['单据编号'],order_2490_for_test,stock)
stop3=datetime.now()
print('单进程用时：',stop3-start3)