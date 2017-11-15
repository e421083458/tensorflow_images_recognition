#!/usr/bin/python
# -*- coding:utf-8 -*-

import urllib
import urllib2
import re
import os
import json
import sys
import thread
import time
from PIL import Image
# from pypinyin import pinyin, lazy_pinyin
reload(sys)
sys.setdefaultencoding('utf8')

# 抓取GG
class SpiderDogLabel:

    # 页面初始化
    def __init__(self):
        self.dog_num = 0
        self.image_num = 0
        self.dog_label = []
        self.siteURL = 'https://sp0.baidu.com/8aQDcjqpAAV3otqbppnN2DJv/api.php?format=json&ie=utf-8&oe=utf-8&query=%E7%8B%97&resource_id=6829&rn=12&from_mid=1&pn={}&type_size=&type_func=&type_feat=&t=1510407462782&cb=jQuery1102040905671791487075_1510399897804&_=1510399897906'

    # 获取索引页面的内容
    def getPage(self, pageIndex):
        url = self.siteURL.format((pageIndex-1)*12)
        request = urllib2.Request(url)
        response = urllib2.urlopen(request)
        return response.read().decode('utf-8')

    # 获取索引界面所有GG的信息，list格式
    def getContents(self, pageIndex):
        page = self.getPage(pageIndex)
        page = str(page)
        start_index = page.find("(")
        end_index = page.rfind(")")
        page = page[start_index+1:end_index]
        dict = json.loads(page, encoding="utf8")
        dog_list = dict["data"][0]["disp_data"]
        contents = []
        for dog_info in dog_list:
            contents.append(dog_info["name"])
        return contents

    # 获取GG个人详情页面
    def getDetailPage(self, infoURL):
        # url = infoURL
        # user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.75 Safari/537.36'
        # Pragma = 'no-cache'
        # Insecure = '1'
        # Accept = 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8'
        # Cookie = 'BDqhfp=%E6%B3%A2%E5%B0%94%E5%A4%9A%E7%8A%AC%26%26NaN-1undefined%26%265802%26%267; BIDUPSID=6D1479598836603133AD496811C631ED; PSTM=1492789346; __cfduid=d93cc59d060d18129d780010a922b0c6c1507362950; userid=51356581; token_=771633c4484C4E484B48454C1508417620903aa336339a2ac4cb1313e6537c15; Hm_lvt_b0e17e90eff522755fac9e19f71a97f7=1508836625; shituhistory=%7B%220%22%3A%22http%3A%2F%2Fb.hiphotos.baidu.com%2Fimage%2Fpic%2Fitem%2F279759ee3d6d55fb3b22e27366224f4a21a4ddfb.jpg%22%2C%221%22%3A%22http%3A%2F%2Fd.hiphotos.baidu.com%2Fimage%2Fpic%2Fitem%2F9a504fc2d56285353eff46749bef76c6a7ef634c.jpg%22%7D; MCITY=-131%3A; BAIDUID=8C7FD4A6735DAF3DF5A2EF2185B77C63:FG=1; tip_show_limit=3; Hm_lvt_9a586c8b1ad06e7e39bc0e9338305573=1508836644,1510285603; indexPageSugList=%5B%22%E8%BF%90%E5%8A%A8%22%2C%22%E8%88%94%E5%B1%8F%22%2C%22%E7%99%BB%E5%B1%B1%22%5D; cleanHistoryStatus=0; firstShowTip=1; PSINO=1; H_PS_PSSID=1453_21095_17001_24880_22158; BDORZ=B490B5EBF6F3CD402E515D22BCDA1598'
        # Connection = 'keep-alive'
        # headers = {'User-Agent': user_agent,'Pragma': Pragma,'Upgrade-Insecure-Requests': Insecure,'Accept': Accept,'Cookie': Cookie,'Connection': Connection,}
        # request = urllib2.Request(url, "", headers)
        # response = urllib2.urlopen(request)
        # page = response.read()
        # json.loads(page.decode('utf-8','ignore'))
        # sys.exit(1)
        response = urllib2.urlopen(infoURL,timeout=3)
        return response.read().decode('utf-8',"ignore")


    # 保存多张写真图片
    def saveImgs(self, images, page, name):
        number = 1
        print u"发现", name, u"共有", len(images), u"张照片"
        print images
        for imageURL in images:
            imageURL = str(imageURL)
            splitPath = imageURL.split('.')
            fTail = splitPath.pop()
            if len(fTail) > 3:
                fTail = "jpg"
            #fileName = name + "/image_" + str(self.image_num) + "." + fTail
            fileName = name + "/image_" + str(number+(page)*30) + "." + fTail
            self.image_num = self.image_num+1
            thread.start_new_thread(self.saveImg, (imageURL, fileName))
            # self.saveImg(imageURL, fileName)
            number += 1

    # 传入图片地址，文件名，保存单张图片
    def saveImg(self, imageURL, fileName):
        url = imageURL
        user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.75 Safari/537.36'
        Pragma = 'no-cache'
        Insecure = '1'
        Encoding = 'gzip, deflate'
        Language = 'zh-CN,zh;q=0.9,en;q=0.8'
        Host = 'img1.imgtn.bdimg.com'
        Accept = 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8'
        Cookie = 'BDqhfp=%E6%B3%A2%E5%B0%94%E5%A4%9A%E7%8A%AC%26%26NaN-1undefined%26%265802%26%267; BIDUPSID=6D1479598836603133AD496811C631ED; PSTM=1492789346; __cfduid=d93cc59d060d18129d780010a922b0c6c1507362950; userid=51356581; token_=771633c4484C4E484B48454C1508417620903aa336339a2ac4cb1313e6537c15; Hm_lvt_b0e17e90eff522755fac9e19f71a97f7=1508836625; shituhistory=%7B%220%22%3A%22http%3A%2F%2Fb.hiphotos.baidu.com%2Fimage%2Fpic%2Fitem%2F279759ee3d6d55fb3b22e27366224f4a21a4ddfb.jpg%22%2C%221%22%3A%22http%3A%2F%2Fd.hiphotos.baidu.com%2Fimage%2Fpic%2Fitem%2F9a504fc2d56285353eff46749bef76c6a7ef634c.jpg%22%7D; MCITY=-131%3A; BAIDUID=8C7FD4A6735DAF3DF5A2EF2185B77C63:FG=1; tip_show_limit=3; Hm_lvt_9a586c8b1ad06e7e39bc0e9338305573=1508836644,1510285603; indexPageSugList=%5B%22%E8%BF%90%E5%8A%A8%22%2C%22%E8%88%94%E5%B1%8F%22%2C%22%E7%99%BB%E5%B1%B1%22%5D; cleanHistoryStatus=0; firstShowTip=1; PSINO=1; H_PS_PSSID=1453_21095_17001_24880_22158; BDORZ=B490B5EBF6F3CD402E515D22BCDA1598'
        Connection = 'keep-alive'
        headers = {'Accept-Encoding': Encoding,'Accept-Language': Language,'Host': Host,'User-Agent': user_agent,'Pragma': Pragma,'Upgrade-Insecure-Requests': Insecure,'Accept': Accept,'Connection': Connection,}
        request = urllib2.Request(url, None, headers)
        u = urllib2.urlopen(request)
        data = u.read()
        f = open(fileName, 'wb')
        f.write(data)
        print u"正在悄悄保存她的一张图片为", fileName
        f.close()

        #图片模式转换
        im = Image.open(fileName)
        # im = im.convert("RGB")
        # print im.size
        # print im.size[0]
        scale = float(im.size[0]/float(im.size[1]))
        if scale<=1:
            width = 500
            height = int(width/scale)
        else:
            height = 500
            width = int(height*scale)
        # print im.size[0]
        im.resize((width,height)).convert("RGB").save(fileName)
        im.close()

    # 创建新目录
    def mkdir(self, path):
        path = path.strip()
        # 判断路径是否存在
        # 存在     True
        # 不存在   False
        isExists = os.path.exists(path)
        # 判断结果
        if not isExists:
            # 如果不存在则创建目录
            print u"偷偷新建了名字叫做", path, u'的文件夹'
            # 创建目录操作函数
            os.makedirs(path)
            return True
        else:
            # 如果目录存在则不创建，并提示目录已存在
            print u"名为", path, '的文件夹已经创建成功'
            return False

    # 将一页GG的信息保存起来
    def savePageInfo(self, pageIndex):
        # 获取第一页GG列表
        contents = self.getContents(pageIndex)
        for item in contents:
            detailURL = "http://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&ct=201326592&is=&fp=result&queryWord+=&cl=2&lm=-1&ie=utf-8&oe=utf-8&adpicid=&st=&word={}&z=&ic=&s=&se=&tab=&width=&height=&face=&istype=&qc=&nc=1&fr=&step_word={}&pn={}&rn=30&gsm=1e&1510412163466="
            # 得到个人详情页面代码
            for i in range(3):
                try:
                    #item = "波尔多犬"
                    test1 = item.encode('utf-8')
                    #print item
                    #print test1
                    test1_1 = urllib.quote(test1)
                    detailURL2 = detailURL.format(test1_1, test1_1, i*30)
                    print u"正在偷偷寻找第", i+1, u"页，看看gougou们在不在"
                    detailPage = self.getDetailPage(detailURL2)
                    dict = json.loads(detailPage)
                    image_list = dict["data"]
                    contents = []
                    for j in range(len(image_list)-1):
                        contents.append(image_list[j]["thumbURL"])
                    #pylist = lazy_pinyin(u""+test1)
                    #pyname = "_".join(pylist)
                    pyname = test1
                    #pyname = str(self.dog_num)
                    if len(contents)>0:
                        self.mkdir("dog_images/jpg/" + pyname)
                        self.saveImgs(contents, i, "dog_images/jpg/" + pyname)
                        time.sleep(2)
                except Exception, arg:
                    print "Error"
                    print arg

            self.dog_label.append(item)  # 设置够够标签
            self.dog_num = self.dog_num + 1 # 类加

    # 传入起止页码，获取GG图片
    def savePagesInfo(self, start, end):
        for i in range(start, end + 1):
            print u"正在偷偷寻找第", i, u"个地方，看看GG们在不在"
            self.savePageInfo(i)


# 传入起止页码即可，在此传入了2,10,表示抓取第2到10页的GG
spider = SpiderDogLabel()
spider.savePagesInfo(1, 1)
# print "dog_label len=%d" % (len(spider.dog_label))
# for i in range(len(spider.dog_label)):
#     print("dog_label[%d]=\"%s\""%(i,spider.dog_label[i]))
# print self.dog_label
