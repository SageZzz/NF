# -*- coding: utf-8 -*-
import datetime

import matplotlib.pyplot as plt  # 导入绘图模块
import numpy as np  # 导入数值计算模块
import pandas as pd  # 导入数据处理模块
import psycopg2
from scipy.optimize import curve_fit  # 导入拟合模块
from sklearn.metrics import r2_score

import xlrd

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)

# postgres config
postgres_host = "114.253.8.150"  # 数据库地址
postgres_port = "6069"  # 数据库端口
postgres_user = "postgres"  # 数据库用户名
postgres_password = "solution"  # 数据库密码
postgres_database = "gridmonitor2018"  # 数据库名字
postgres_table = ""  # 数据库中的表的名字

conn_string = "host=" + postgres_host + " port=" + postgres_port + " dbname=" + postgres_database + \
              " user=" + postgres_user + " password=" + postgres_password

device_id = [['铁营镇孵化园区站', '9500H171100048']]

""" 
桓台微站 [['城区街办', '9100E180300064', '9100E180300066', '9100E180300069', '9100E171200071', '9100E180300075',
              '9100E180300076', '9100E180400093', '9100E180400102'],
             ['果里镇', '9100E180300072',  '9100E180400082', '9100E180400085', '9100E180400087', '9100E180400088',
              '9100E180400090', '9100E180400092', '9100E180400097', '9100E180400099', '9100E180400100',
              '9100E180400104', '9100E180400105', '9100E180400114'],
             ['锦秋', '9100E180300061', '9100E180300073', '9100E180400080', '9100E180400086', '9100E180400095', '9100E180400096',
              '9100E180400103', '9100E180400106', '9100E180400107', '9100E180400108', '9100E180400115'],
             ['新城镇', '9100E180300065', '9100E180300074', '9100E180400094', '9100E180400110'],
             ['马桥镇', '9100E180300060', '9100E180300062', '9100E180400078', '9100E180400098'],
             ['起凤镇', '9100E180300067', '9100E180400081', '9100E180400089']]
深圳天台 ['9100E180400116', '9100E190600015', '9100F191200147']
"""


color_pad = [['#FFB6C1', '#FFC0CB', '#DC143C', '#DB7093', '#FF69B4', '#FF1493', '#C71585', '#DA70D6', '#D8BFD8',
              '#DDA0DD', '#EE82EE', '#FF00FF', '#FF00FF', '#8B008B', '#800080', '#BA55D3'],
             ['#9400D3', '#9932CC', '#4B0082', '#8A2BE2', '#9370DB', '#7B68EE', '#6A5ACD', '#483D8B', '#0000FF',
              '#0000CD', '#191970', '#00008B', '#000080', '#4169E1', '#6495ED', '#B0C4DE'],
             ['#778899', '#708090', '#1E90FF', '#4682B4', '#87CEFA', '#87CEEB', '#00BFFF', '#ADD8E6', '#B0E0E6',
              '#5F9EA0', '#AFEEEE', '#00FFFF', '#00FFFF', '#00CED1', '#2F4F4F', '#008B8B'],
             ['#008080', '#48D1CC', '#20B2AA', '#40E0D0', '#7FFFAA', '#00FA9A', '#00FF7F', '#3CB371', '#2E8B57',
              '#90EE90', '#98FB98', '#8FBC8F', '#32CD32', '#00FF00', '#228B22', '#008000'],
             ['#006400', '#7FFF00', '#7CFC00', '#ADFF2F', '#556B2F', '#F5F5DC', '#FAFAD2', '#FFFFE0', '#FFFF00',
              '#808000', '#BDB76B', '#FFFACD', '#EEE8AA', '#F0E68C', '#FFD700', '#DAA520'],
             ['#F5DEB3', '#FFE4B5', '#FFA500', '#FFEFD5', '#FFEBCD', '#FFDEAD', '#FAEBD7', '#D2B48C', '#DEB887',
              '#FFE4C4', '#FF8C00', '#CD853F', '#FFDAB9', '#F4A460', '#D2691E', '#8B4513'],
             ['#A0522D', '#FFA07A', '#FF7F50', '#FF4500', '#E9967A', '#FF6347', '#FA8072', '#F08080', '#BC8F8F',
              '#CD5C5C', '#FF0000', '#A52A2A', '#B22222', '#8B0000', '#800000', '#DCDCDC'],
             ['#D3D3D3', '#C0C0C0', '#A9A9A9', '#808080', '#696969', '#000000']]

time_duty = [['2020-08-18 17', '2020-08-21 15'], ['2020-08-18 17', '2020-08-21 15']]

labels = ['CO', 'NO2', 'O3', 'SO2', 'PM25', 'PM10']

diff = []
target = []
mlen = 0
paraDf = pd.DataFrame(columns=('a', 'b', 'c', 'd', 'e', 'r2'), index=labels)

func = [
    (lambda x, a, b, c, d, e : a * x + b * diff[7][0:mlen] + c * 0 + d*0 + e),  # co
    (lambda x, a, b, c, d, e : a * x + b * 0 + c * diff[6][0:mlen] + d*0 + e),  #no2
    (lambda x, a, b, c, d, e : a * x + b * diff[1][0:mlen] + c * 0 + d * 0 + e),  #o3
    (lambda x, a, b, c, d, e : a * x + b * diff[1][0:mlen] + c * 0 + d * 0 + e),  #so2
#    (lambda x, a, b, c, d, e : a*0 + c * x + b * diff[8][0:mlen] + d*0+ e),  #pm
#    (lambda x, a, b, c, d, e : a*0 + c * x + b * diff[8][0:mlen] + d*0+ e)  #pm
    (lambda x, a, b, c, d, e: (x*c+d)/(a+b*(diff[8][0:mlen] * diff[8][0:mlen])/(1-diff[8][0:mlen]))+e),  # pm
    (lambda x, a, b, c, d, e: (x*c+d)/(a+b*(diff[8][0:mlen] * diff[8][0:mlen])/(1-diff[8][0:mlen]))+e)  # pm
]

param_bounds = [
    # a           b             c             d        e
    ([0, -1000, -1000, -1000, -1000],      [0.1, 1000, 1000, 1000, 1000]),
    ([-1000, -1000, -1000, -1000, -1000],      [1000, 1000, 1000, 1000, 1000]),
    ([-1000, -1000, -1000, -1000, -1000],      [1000, 1000, 1000, 1000, 1000]),
    ([-1000, -1000, -1000, -1000, -1000],      [1000, 1000, 1000, 1000, 1000]),
    ([-1000, -1000, 0.7, -1000, -1000],      [1000, 1000, 2, 1000, 1000]),
    ([-1000, -1000, 0.7, -1000, -1000],      [1000, 1000, 2, 1000, 1000]),
]
"""
func = [
    (lambda x, a, b, c, d, e : a * x + b * diff[7] + c * diff[6] + d*0 + e),  # co
    (lambda x, a, b, c, d, e : a * x + b * diff[7] + c * diff[6] + d*0 + e),  #no2
    (lambda x, a, b, c, d, e : a * x + b * diff[1] + c * diff[7] + d * diff[6] + e),  #o3
    (lambda x, a, b, c, d, e : a * x + b * diff[1] + c * diff[7] + d * diff[6] + e),  #so2
    (lambda x, a, b, c, d, e : (c * x + d) / (a + b * diff[8]**2/(1-diff[8])) + e),  #pm
    (lambda x, a, b, c, d, e : (c * x + d) / (a + b * diff[8]**2/(1-diff[8])) + e)  #pm
]
"""



sql_command = '''select monitor_time  monitor_time,device_id,
        to_number(max(case when lower(param_code)='no2_act' then param_value else null end),'99999.99999') no2_act,
        to_number(max(case when lower(param_code)='no2_ref' then param_value else null end),'99999.99999') no2_ref,
        to_number(max(case when lower(param_code)='co_act' then param_value else null end),'99999.99999') co_act,
        to_number(max(case when lower(param_code)='co_ref' then param_value else null end),'99999.99999') co_ref,
        to_number(max(case when lower(param_code)='oz_act' then param_value else null end),'99999.99999') oz_act,
        to_number(max(case when lower(param_code)='oz_ref' then param_value else null end),'99999.99999') oz_ref,
        to_number(max(case when lower(param_code)='so2_act' then param_value else null end),'99999.99999') so2_act,
        to_number(max(case when lower(param_code)='so2_ref' then param_value else null end),'99999.99999') so2_ref,
        to_number(max(case when lower(param_code)='pm25at' then param_value else null end),'99999.99999') PM25,
        to_number(max(case when lower(param_code)='pm10at' then param_value else null end),'99999.99999') PM10,
        to_number(max(case when lower(param_code)='in_temp' then param_value else null end),'99999.99999') in_temp,
        to_number(max(case when lower(param_code)='in_humi' then param_value else null end),'99999.99999')/100 in_humi
    from t_device_data where 1=1
        and monitor_time >= to_timestamp(\'{0[0]}\', 'yyyy-MM-dd hh24')
        and monitor_time <= to_timestamp(\'{0[1]}\', 'yyyy-MM-dd hh24')
        and device_id=\'{1}\'
        group by monitor_time,device_id order by monitor_time asc'''

sql_command_t = '''select monitortime,pointname,pointcode,so2,no2,co,o3,pm10,pm25
    from t_env_airdata_hour_point where 1=1
        and monitortime >= to_timestamp(\'{0[0]}\', 'yyyy-MM-dd hh24')
        and monitortime <= to_timestamp(\'{0[1]}\', 'yyyy-MM-dd hh24')
--		and pointname='华侨城'
--		and pointname='少海街办'
        and pointcode=\'{1}\'
        order by monitortime asc'''


def get_datas_from_sql(sqlcofig: str, ids: list, td: list) -> list:
    datas = []
    connect = psycopg2.connect(sqlcofig)  # 连接sql
    for idd in ids:
        data = []
        for id in idd:
            if id in ['铁营镇孵化园区站', '果里镇', '锦秋', '新城镇', '马桥镇', '起凤镇']:
                target_data = pd.read_excel(r'D:\Users\文档\数据\乐陵\国标2.xls', sheet_name=id,
                                     header=None, names=['time', 'so2', 'no2', 'o3', 'co', 'pm10', 'pm25'],
                                            skiprows=1, skipfooter=0)
                for i in range(len(target_data['time'])):
                    target_data['time'][i] = datetime.datetime.strptime(target_data['time'][i], "%Y年%m月%d日  %H时%M分")
                data.append(target_data)
                print(target_data)
            else:
                cmd = sql_command.format(td[0], id)  # 初始化sql命令
                try:
                    data.append(pd.read_sql(cmd, connect))
                except:
                    print("{} load data from postgres failure !".format(id))
                    exit()
            print("{} load sql success !".format(id))
        datas.append(data)
    connect.close()
    return datas


def fit_and_draw_data_to_fig(idd, dat_t, dat, color) -> None:
    if idd in ['铁营镇孵化园区站', '果里镇', '锦秋', '新城镇', '马桥镇', '起凤镇']:
        dat_t['time'] = pd.to_datetime(dat_t['time'])
        dat_t[['so2', 'no2', 'co', 'o3', 'pm25', 'pm10']] = dat_t[['so2', 'no2', 'co', 'o3', 'pm25', 'pm10']].apply(pd.to_numeric, errors='coerce')
        dat_t = dat_t.groupby([dat_t['time'].dt.day, dat_t['time'].dt.hour], as_index=True).mean()  # 小时平均值
        global target
        target = [
             dat_t['co'],
             dat_t['no2'],
             dat_t['o3'],
             dat_t['so2'],
            dat_t['pm25'][1:],  # 错峰2小时
            dat_t['pm10'][1:]  # 错峰2小时
        ]
        for i in range(0, 6):
            axs[i].plot(np.arange(0, len(target[i])), target[i], 'b', label='{0} {1}'.format(idd, labels[i]))
    else:
        dat['monitor_time'] = pd.to_datetime(dat['monitor_time'])
        dat = dat.groupby([dat['monitor_time'].dt.day, dat['monitor_time'].dt.hour], as_index=True).mean()  # 小时平均值
        global diff
        diff = [
            (dat['co_act'] - dat['co_ref'])*1000,
            (dat['no2_act'] - dat['no2_ref'])*1000,
            (dat['oz_act'] - dat['oz_ref'])*1000,
            (dat['so2_act'] - dat['so2_ref'])*1000,
            dat['pm25'][:-1],
            dat['pm10'][:-1],
            dat['in_temp'],
            dat['in_humi'],
            dat['in_humi']
                ]
        for i in range(0, 6):
            global mlen
            mlen = min(len(diff[i]), len(target[i]))
            popt, pcov = curve_fit(func[i], diff[i][0:mlen], target[i][0:mlen], maxfev=100000)  #, bounds=param_bounds[i])
#            print(popt,pcov)
            for j in range(len(popt)):
              if popt[j] == 1:
                    popt[j] = 0
            yval = func[i](diff[i][0:mlen], *popt)
            r2 = r2_score(target[i][0:mlen], yval)
            axs[i].plot(np.arange(0, len(diff[i][0:mlen])), yval, color=color,
                        label='%s %s %.4f,%.4f,%.4f,%.4f,%.4f,r2:%f' % (idd, labels[i], popt[0], popt[1], popt[2], popt[3], popt[4], r2))
            paraDf.loc[labels[i]] = [popt[0], popt[1], popt[2], popt[3], popt[4], r2]
            print("{} fit success !".format(idd))


if __name__ == "__main__":#  绘图

    fig, ax = plt.subplots(3, 2, figsize=(24, 16))
    axs = ax.flatten()
    sqlDatas = get_datas_from_sql(conn_string, device_id, time_duty)
    writer = pd.ExcelWriter(r'D:\Python Projects\EXCEL\fit_%s.xlsx' % (datetime.datetime.now().strftime("%Y%m%d%H%M")))
    for x in range(len(sqlDatas)):
        for i in range(len(sqlDatas[x][0])):
            print(x, i)
            if sqlDatas[x][0].loc[i, 'co'] == 0:
                sqlDatas[x][0].loc[i, 'co'] = sqlDatas[x][0].loc[i - 1 if i - 1 > 0 else i + 1, 'co']
            if sqlDatas[x][0].loc[i, 'no2'] == 0:
                sqlDatas[x][0].loc[i, 'no2'] = sqlDatas[x][0].loc[i - 1 if i - 1 > 0 else i + 1, 'no2']
            if sqlDatas[x][0].loc[i, 'so2'] == 0:
                sqlDatas[x][0].loc[i, 'so2'] = sqlDatas[x][0].loc[i - 1 if i - 1 > 0 else i + 1, 'so2']
            if sqlDatas[x][0].loc[i, 'o3'] == 0:
                sqlDatas[x][0].loc[i, 'o3'] = sqlDatas[x][0].loc[i - 1 if i - 1 > 0 else i + 1, 'o3']
            if sqlDatas[x][0].loc[i, 'pm25'] == 0:
                sqlDatas[x][0].loc[i, 'pm25'] = sqlDatas[x][0].loc[i - 1 if i - 1 > 0 else i + 1, 'pm25']
            if sqlDatas[x][0].loc[i, 'pm10'] == 0:
                sqlDatas[x][0].loc[i, 'pm10'] = sqlDatas[x][0].loc[i - 1 if i - 1 > 0 else i + 1, 'pm10']

        for i in range(0, len(device_id[x])):
            fit_and_draw_data_to_fig(device_id[x][i], sqlDatas[x][0], sqlDatas[x][i], color_pad[x][i])
            if i > 0:
                sheetname = device_id[x][i]
                paraDf.to_excel(writer, sheet_name=sheetname)
    writer.save()
    font = {'family' : 'SimHei', 'weight' : 'normal', 'size' : 8}

    for a in axs:
        a.legend(loc='upper left', prop=font)

    plt.tight_layout(pad=2, w_pad=2, h_pad=5)

    plt.savefig(r'D:\Python Projects\PNG\fit_%s.png' % (datetime.datetime.now().strftime("%Y%m%d%H%M")))
    plt.show()
