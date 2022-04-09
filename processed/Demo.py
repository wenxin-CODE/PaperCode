#!/usr/bin/python
# -*- coding: utf-8 -*-
from tkinter import messagebox
from tkinter import *
import tkinter.font as tkFont
from tkinter import filedialog
import tkinter.messagebox
import cv2

root = tkinter.Tk()
ft2 = tkFont.Font(family='Microsoft YaHei', size=20, weight=tkFont.BOLD, underline=0, overstrike=0)


def set_win_center(root, curWidth='', curHight=''):
    '''
    设置窗口大小，并居中显示
    :param root:主窗体实例
    :param curWidth:窗口宽度，非必填，默认200
    :param curHight:窗口高度，非必填，默认200
    :return:无
    '''
    if not curWidth:
        '''获取窗口宽度，默认200'''
        curWidth = root.winfo_width()
    if not curHight:
        '''获取窗口高度，默认200'''
        curHight = root.winfo_height()
    # print(curWidth, curHight)

    # 获取屏幕宽度和高度
    scn_w, scn_h = root.maxsize()
    # print(scn_w, scn_h)

    # 计算中心坐标
    cen_x = (scn_w - curWidth) / 2
    cen_y = (scn_h - curHight) / 2
    # print(cen_x, cen_y)

    # 设置窗口初始大小和位置
    size_xy = '%dx%d+%d+%d' % (curWidth, curHight, cen_x, cen_y)
    root.geometry(size_xy)


def openfile():
    def is_inside(o, i):
        ox, oy, ow, oh = o
        return ox > ix and oy > iy and ox + ow < ix + iw and oy + oh < iy + ih

    def draw_person(image, person):
        x, y, w, h = person
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)

    path = filedialog.askopenfilename(title='打开', filetypes=[('S2out', '*.*'), ('All Files', '*')])
    print(path)
    img = cv2.imread(path)
    # a, b = img.shape[:2]
    # img = cv2.resize(img, (a//5, b//5))
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    found, w = hog.detectMultiScale(img)

    found_filtered = []
    l = len(w)
    print(l)

    for ri, r in enumerate(found):
        for qi, q in enumerate(found):
            if ri != qe and is_inside(r, q):
                break
            else:
                found_filtered.append(r)
    for person in found_filtered:
        draw_person(img, person)
    cv2.imshow('people detection', img)

    if l >= 3:
        Label(root, text='那儿有3个人以上', font=ft2).pack(padx=0, pady=20)
        Label(root, text=('总共有', l, '个人'), font=ft2).pack(padx=0, pady=20)
        root.mainloop()
    else:
        Label(root, text='那儿有少于2个人', font=ft2).pack(anchor=CENTER)
        Label(root, text=('总共有', l, '个人'), font=ft2).pack(padx=0, pady=20)
        root.mainloop()

    cv2.waitKey(0)
    cv2.destroyAllWindows()


root.title("图像人数统计")
root.geometry('500x350+500+200')
btn2 = tkinter.Button(root, text='图像人数统计', font=('microsoft yahei', 14, ''), width=10, height=2, command=openfile)
btn3 = tkinter.Button(root, text='退出', font=('microsoft yahei', 14, ''), width=10, height=2, command=root.quit)
btn2.pack()
btn3.pack()

root.mainloop()







