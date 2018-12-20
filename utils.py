#!/usr/bin/python
#coding:utf-8

from PIL import Image

def get_img_info(self,image_path):
  """获取图像信息，为read_and_decode函数提供参数，若已知图像格式则不需调用
    PIL中图像mode有9种，1/L/P/RBG/RBGA/CMYK/YCbCr/I/F
    在此处仅介绍常用的三种，其余请自行查询
    L：8位像素灰度图，1通道
    RGB：3x8位像素彩色图，3通道
    RGBA：4x8位像素彩色图+透明通道，4通道
  Args:
    image_path: string, 选定图像的路径
  Return: 
    image_height: int, 图像高度
    image_width: int, 图像宽度
    image_mode: string, 图像模式   
  """
  image = Image.open(image_path)
  image_height = image.size[0]
  image_width = image.size[1]
  image_mode = image.mode
  return  image_height,image_width,image_mode
