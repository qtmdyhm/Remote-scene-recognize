我不知道为啥预览模式的自述文件很乱，你可以使用code模式更直观的查看它
这是一个用于遥感图文匹配的系统，支持图像类别预测、图像目标计数以及场景检索
该项目具有以下结构：
│  app.py
│  classify.py
│  count.py
│  requirements.txt
│  search.py
│
├─models
│      RemoteCLIP-ViT-L-14.pt
│
├─static
│  ├─results
│  │      .gitkeep
│  │
│  └─uploads
│          .gitkeep
│
├─templates
│      index.html
│      login.html
│
└─__pycache__
        classify.cpython-39.pyc
        count.cpython-39.pyc
        search.cpython-39.pyc
        
现在请按照下面的方法运行它：
1.创建一个空文件夹
2.用git打开这个文件夹执行下面命令：
git clone https://github.com/qtmdyhm/Remote-scene-recognize.git
3.点击下面链接下载模型权重，并且把模型权重放到models文件夹里：
https://drive.google.com/file/d/1ifA2z1qkvYqRv2MRraci83wDUo7Cb3Kj/view?usp=drive_link
4.启动你的终端，执行下面命令创建项目环境：
conda create -n remote python=3.9
5.激活项目环境
conda activate remote
6.切换终端里的文件路径到你克隆的文件夹(Remote-scene-recognize)：
cd 你的文件夹地址（记得删除地址的引号）
6.安装依赖的包：
pip install -r requirements.txt
7.运行：
python app.py

按照上述方法运行后你将会看到类似examples里的运行截图

写在最后：
1.app.py里注释掉的部分用于将本地服务渗透到公网，使别的设备也能访问，如果你有这个需求，你可以参考pyngrok
2.模型权重来自https://link.zhihu.com/?target=https%3A//github.com/ChenDelong1999/RemoteCLIP           我不生产模型，我只是模型的搬运工
3.尽可能的使用英文进行图文匹配，如果使用中文你将会看到模型性能大大下降
