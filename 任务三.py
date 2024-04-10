import numpy as np
import pandas as pd
import netCDF4 as nc
import xarray as xr

import math

import matplotlib
import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib.transforms as mtransforms
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

import cnmaps
from cnmaps import get_adm_maps, draw_map
from cnmaps.sample import load_dem

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.feature import NaturalEarthFeature
import cartopy.io.shapereader as shpreader

import shapefile
import geopandas as gpd
from shapely import geometry
import os

def adjust_sub_axes(ax_main, ax_sub, shrink):
    '''
    将ax_sub调整到ax_main的右下角. shrink指定缩小倍数.
    当ax_sub是GeoAxes时, 需要在其设定好范围后再使用此函数.
    '''
    bbox_main = ax_main.get_position()
    bbox_sub = ax_sub.get_position()
    # 使shrink=1时ax_main与ax_sub等宽或等高.
    if bbox_sub.width > bbox_sub.height:
        ratio = bbox_main.width / bbox_sub.width * shrink
    else:
        ratio = bbox_main.height / bbox_sub.height * shrink
    wnew = bbox_sub.width * ratio
    hnew = bbox_sub.height * ratio
    bbox_new = mtransforms.Bbox.from_extents(bbox_main.x1 - wnew, bbox_main.y0, bbox_main.x1, bbox_main.y0 + hnew)
    ax_sub.set_position(bbox_new)

shp_path = r'C:\Users\28166\Desktop\china\china.shp'

# 设置标题和字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14

# # scatter数据处理
# file2 = r'D:\huankepeixun\mission 2\mission 2 data-DailyConc_China_CNEMC_EachStation_2013to2021_nc\Daily_OBS_Concs4EachStation_Y2013toY2021.nc'
# dataset2 = nc.Dataset(file2)
# # dict_keys(['char_year', 'char_species', 'char_std_index', 'std_lat', 'std_lon', 'daily_conc'])9，366，8，1674
# lats2 = dataset2.variables['std_lat'][:]
# lons2 = dataset2.variables['std_lon'][:]
# concs2 = dataset2.variables['daily_conc'][0:8, :, 0, :]
# lats_data2 = np.array(lats2)
# lons_data2 = np.array(lons2)
# concs_data2 = np.array(concs2)
# concs_data2 = np.where(concs_data2 == -999, np.nan, concs_data2)
# sub_arrays = np.split(concs_data2, 8, axis=0)
# all_data = []
# reshaped_subarrays = [sub_array.reshape(366, 1674) for sub_array in sub_arrays]  # 将8个(1,366,1674)转换成9个(366,1674)
# for i, sub_array in enumerate(reshaped_subarrays):
#     last_list = [np.nanmean(x) for x in zip(*sub_array)]  # 在每一个（366,1674）跑的过程中，计算每个站点的年平均，即二维按列求和
#     all_data.append(last_list)
# all_data = np.array(all_data)
# print(all_data.shape)

# contourf数据处理
file3 = r'D:\huankepeixun\mission 3\TAP_Daily_PM25_Y2001toY2020\TAP_Daily_PM25_Y2001toY2020.nc'
dataset3 = nc.Dataset(file3)
# dict_keys(['latitude', 'longitude', 'Daily_PM25'])
# latitude(450),Latitude from 15.05 to 59.95 by 0.1
# longitude(700),Longitude from 70.05 to 139.95 by 0.1
# float32 Daily_PM25(year, day, lat, lon)(20, 366, 450, 700)
lats3 = dataset3.variables['latitude'][:]
lons3 = dataset3.variables['longitude'][:]
lats_data3 = np.array(lats3)
lons_data3 = np.array(lons3)
concs3 = dataset3.variables['Daily_PM25'][12:20, :, :, :]
concs3 = np.array(concs3)
concs3 = np.where(concs3 == -999, np.nan, concs3)
data3 = xr.DataArray(concs3, dims=("year", "day", "lasts", "lons"))
data3 = data3.loc[:, :, :, :].mean("day")
print(data3.shape)
print(data3)

#  画图
proj = ccrs.PlateCarree()  # 创建地图投影
fig = plt.figure(figsize=(33, 20))
fig.suptitle('Space Distribution of PM2.5', fontsize=20, weight='bold', y=0.94)  # 设置大图标题,y用来控制大标题的相对位置
axes_main = fig.subplots(3, 3, subplot_kw=dict(projection=proj))
axes_sub = fig.subplots(3, 3, subplot_kw=dict(projection=proj))

extent_main = [70, 140, 15, 55]
extents_sub = [105, 125, 0, 25]

fig.subplots_adjust(right=0.85)  # 设置色条

colorlevel = [0,5,10,20,30,40,60,80,100,120,150,200,240]
colordict = ['#FFFFFF', '#C2E8FA', '#86C5EB', '#5196CF', '#49A383', '#6ABF4A',  '#D9DE58', '#F8B246','#F26429', '#DD3528', '#BC1B23', '#921519']
color_map = mcolors.ListedColormap(colordict)
norm = mcolors.BoundaryNorm(colorlevel, 12)

china = gpd.read_file('D:/各类文件/china_map/china.shp')
nanhai = gpd.read_file('D:/各类文件/china_map/9duanxian/9duanxian.shp')
shengji = gpd.read_file('D:/各类文件/china_map/分省各级别shp（超全）/2. Province/province.shp')
n = 2013
k = 0
for i in range(3):
    for j in range(3):
        if i == 2 and j == 2:
            # 添加颜色条
            axes_main[i, j].axis('off')
            axes_sub[i, j].axis('off')
        else:
            axes_main[i, j].set_extent(extent_main, crs=proj)
            china.plot(ax=axes_main[i, j], color='white', edgecolor='k', zorder=2)
            shengji.plot(ax=axes_main[i, j], color='None', edgecolor='gray', linewidths=0.3, zorder=4)
            # 绘制填色图
            sc = axes_main[i, j].contourf(lons3, lats3, data3.loc[k, :, :], cmap=color_map, norm=norm,
                                          levels=[0, 5, 10, 20, 30, 40, 60, 80, 100, 120, 150, 200, 240],
                                          transform=proj, zorder=3)
            # 绘制散点图
            # sc2 = axes_main[i, j].scatter(lons2, lats2, c=all_data[k], s=12, edgecolor='k', linewidths=0.1,
            #                               cmap=color_map, norm=norm, transform=proj, zorder=5)
            # 设置经纬度
            gl = axes_main[i, j].gridlines(crs=proj, draw_labels=True, linestyle=":", linewidth=0.1, x_inline=False,
                                           y_inline=False, color='k', alpha=0.5, xlines=False, ylines=False)
            gl.top_labels = False
            gl.right_labels = False
            gl.xformatter = LONGITUDE_FORMATTER  # x轴设为经度的格式
            gl.yformatter = LATITUDE_FORMATTER  # y轴设为纬度的格式
            gl.xlocator = mticker.FixedLocator([80, 90, 100, 110, 120, 130, 140])  # extent[0], extent[1]+0.5, 10
            gl.ylocator = mticker.FixedLocator([20, 30, 40, 50, 60])  # extent[2], extent[3]+0.5, 10
            gl.xlines = False
            gl.ylines = False
            font2 = {'size': 10, 'family': 'Times New Roman', 'weight': 'normal'}
            gl.xlabel_style = font2
            gl.ylabel_style = font2
            # 添加年份文字
            axes_main[i, j].text(0.03, 0.91, f'{n}', bbox={'facecolor': 'white', 'alpha': 1}, fontsize=8, transform=axes_main[i, j].transAxes)
            # 画南海及九段线
            axes_sub[i, j].set_extent(extents_sub, crs=proj)
            china.plot(ax=axes_sub[i, j], color='white', edgecolor='gray', zorder=0,linewidths=0.35)
            nanhai.plot(ax=axes_sub[i, j], color='gray', edgecolor='gray', zorder=1)
            adjust_sub_axes(axes_main[i, j], axes_sub[i, j], shrink=0.3)
            cbar_ax = fig.add_axes([0.88, 0.15, 0.01, 0.7])  # 位置
            cbar = fig.colorbar(mappable=sc, cax=cbar_ax, format='%.2f', shrink=0.88,
                                ticks=[0, 5, 10, 20, 30, 40, 60, 80, 100, 120, 150, 200, 240])
            ax0 = cbar.ax  # 将colorbar变成一个新的ax对象，可通过ax对象的各种命令来调整colorbar
            ax0.set_title('Concs', fontproperties='Times New Roman', weight='normal', size=15, pad=20)
            ax0.tick_params(which='major', direction='in', labelsize=12, length=7.5)
        n = n + 1
        k = k + 1
plt.show()












































