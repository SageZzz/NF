# -*- coding: utf-8 -*-
import datetime
import pandas as pd  # 导入数据处理模块

writer = pd.ExcelWriter(r'D:\Python Projects\EXCEL\国标2.xlsx')
target_data = pd.read_excel(r'D:\Users\文档\数据\乐陵\国标2.xls',
                            header=None, names=['time', 'so2', 'no2', 'o3', 'co', 'pm10', 'pm25', 'temp', 'humi'],
                            skiprows=1, skipfooter=0)
#
for i in range(len(target_data['time'])):
    target_data['time'][i] = datetime.datetime.strptime(target_data['time'][i], "%Y年%m月%d日  %H时%M分")

target_data['time'] = pd.to_datetime(target_data['time'])
target_data[['so2', 'no2', 'co', 'o3', 'pm25', 'pm10']] = target_data[['so2', 'no2', 'co', 'o3', 'pm25', 'pm10']].apply(pd.to_numeric, errors='coerce')
target_data = target_data.groupby([target_data['time'].dt.day, target_data['time'].dt.hour], as_index=True).mean()  # 小时平均值

print(target_data)
target_data.to_excel(writer, sheet_name='sheel1')
writer.save()