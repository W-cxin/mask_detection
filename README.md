# mask_detection

对是否戴口罩进行检测，分为三类：with_mask，without_mask，mask_weared_incorrect。此处以科大讯飞这个赛题为例：[口罩佩戴识别挑战赛](http://challenge.xfyun.cn/topic/info?type=wear-mask&option=ssgy)，从赛题分析上来看就是一个简单的三分类问题，我采用了两种baseline来做，**第一个baseline用的就是YOLOv5，准确率为0.96984；第二个baseline用的是swin transformer，准确率能更高，达到了0.98175**。

此处对怎么使用YOLOv5进行口罩检测做一个简单叙述，包括主要的流程以及提交的一些准备。

## 1. 数据预处理

**第1步：将图片都转成png格式**

对数据集进行观察，发现数据集里面有jpg和png图片两种形式，为了方便后面的处理，先将所有的图片都转化为png格式：

```python
import os
def Rename(filename):
    transF = os.path.splitext(filename)
    if transF[1] == '.jpg':
        new_name = transF[0] + '.png'
        os.rename(filename, new_name)

def transform(file_path):
    # 读取所有文件
    allFiles = os.listdir(file_path)
    for file in allFiles:
        file_path1 = os.path.join(file_path, file)
        if os.path.isdir(file_path1):
            transform(file_path1)
        else:
            os.chdir(file_path)
            Rename(file)

if __name__ == '__main__':
    # 文件夹路径
    file_path = "/mask_detection/VOCdevkit/VOC2007/JPEGImages/"
    transform(file_path)
```

**第2步：对图片进行txt文件的生成**

首先下载YOLOv5的代码以及预训练模型：[GitHub - ultralytics/yolov5 at v5.0](https://github.com/ultralytics/yolov5/tree/v5.0)，预训练模型使用的是yolov5s.pt。

要将数据集整理成YOLOv5的形式，首先要标注ground truth。YOLOv5的ground truth使用的是txt格式，txt文件中每一行的五个值，分别对应类别，x的中心点，y的中心点，长，宽：

```
<object-class> <x_center> <y_center> <width> <height>
```

考虑到数据集中的每张图片只有一个目标，而且分辨率很低，大部分是128*128，为了方便起见，不进行一一的标注，直接将所有图片的txt文件标成如下形式：

```
0 0.5 0.5 1 1 # 无口罩：0
1 0.5 0.5 1 1 # 有口罩：1
2 0.5 0.5 1 1 # 不规范：2
```

相当于直接把这整张图当成标出的ground truth，具体代码如下：

```python
import os
readpath = '/mask_detection/kdxf-datasets/train/without_mask/' # 文件夹位置
files = os.listdir(readpath) # 读取文件夹下文件名
savepath = '/mask_detection/labels/'
# 无口罩：0，有口罩：1，不规范：2
for file in files: # 遍历所有文件名
    print(file)
    f = open(savepath + '/' + file[:-4] + '.txt', 'w')  # 创建文本文件
    if any(file.endswith(extension) #判断是不是图片
           for extension in ['.tif', '.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']):
        f.write( '0 0.5 0.5 1 1' + '\n') # 写入文件
f.close()
```

**第3步：将yolo格式转成voc**

接下来将数据集划分为训练集和验证集，参考的博客主要是这篇：[数据集格式转化及训练集和验证集划分](https://blog.csdn.net/didiaopao/article/details/120022845?spm=1001.2014.3001.5502)，为了方便进行数据处理，我没有改变文件夹的名称，直接使用的是跟这个博主一样的格式，这个博主提到了一个问题，之前生成的txt文件直接使用的话可能会有数据集加载的错误，按照他的思路我先将yolo的txt标签格式转成了voc的xml的标签格式：

```python
# 第3步：将yolo格式转成voc
from xml.dom.minidom import Document
import os
import cv2

# def makexml(txtPath, xmlPath, picPath):  # txt所在文件夹路径，xml文件保存路径，图片所在文件夹路径
def makexml(picPath, txtPath, xmlPath):  # txt所在文件夹路径，xml文件保存路径，图片所在文件夹路径
    """此函数用于将yolo格式txt标注文件转换为voc格式xml标注文件
    在自己的标注图片文件夹下建三个子文件夹，分别命名为picture、txt、xml
    """
    dic = {'0': "0",  # 创建字典用来对类型进行转换
           '1': "1",  # 此处的字典要与自己的classes.txt文件中的类对应，且顺序要一致
           '2': "2"
           }
    files = os.listdir(txtPath)
    for i, name in enumerate(files):
        xmlBuilder = Document()
        annotation = xmlBuilder.createElement("annotation")  # 创建annotation标签
        xmlBuilder.appendChild(annotation)
        txtFile = open(txtPath + name)
        txtList = txtFile.readlines()
        try:
            img = cv2.imread(picPath + name[:-4] + ".png")
        except AttributeError:
            img = cv2.imread(picPath + name[:-4] + ".jpg")

        Pheight, Pwidth, Pdepth = img.shape

        folder = xmlBuilder.createElement("folder")  # folder标签
        foldercontent = xmlBuilder.createTextNode("driving_annotation_dataset")
        folder.appendChild(foldercontent)
        annotation.appendChild(folder)  # folder标签结束

        filename = xmlBuilder.createElement("filename")  # filename标签
        filenamecontent = xmlBuilder.createTextNode(name[:-4] + ".png")
        filename.appendChild(filenamecontent)
        annotation.appendChild(filename)  # filename标签结束

        size = xmlBuilder.createElement("size")  # size标签
        width = xmlBuilder.createElement("width")  # size子标签width
        widthcontent = xmlBuilder.createTextNode(str(Pwidth))
        width.appendChild(widthcontent)
        size.appendChild(width)  # size子标签width结束

        height = xmlBuilder.createElement("height")  # size子标签height
        heightcontent = xmlBuilder.createTextNode(str(Pheight))
        height.appendChild(heightcontent)
        size.appendChild(height)  # size子标签height结束

        depth = xmlBuilder.createElement("depth")  # size子标签depth
        depthcontent = xmlBuilder.createTextNode(str(Pdepth))
        depth.appendChild(depthcontent)
        size.appendChild(depth)  # size子标签depth结束

        annotation.appendChild(size)  # size标签结束

        for j in txtList:
            oneline = j.strip().split(" ")
            object = xmlBuilder.createElement("object")  # object 标签
            picname = xmlBuilder.createElement("name")  # name标签
            namecontent = xmlBuilder.createTextNode(dic[oneline[0]])
            picname.appendChild(namecontent)
            object.appendChild(picname)  # name标签结束

            pose = xmlBuilder.createElement("pose")  # pose标签
            posecontent = xmlBuilder.createTextNode("Unspecified")
            pose.appendChild(posecontent)
            object.appendChild(pose)  # pose标签结束

            truncated = xmlBuilder.createElement("truncated")  # truncated标签
            truncatedContent = xmlBuilder.createTextNode("0")
            truncated.appendChild(truncatedContent)
            object.appendChild(truncated)  # truncated标签结束

            difficult = xmlBuilder.createElement("difficult")  # difficult标签
            difficultcontent = xmlBuilder.createTextNode("0")
            difficult.appendChild(difficultcontent)
            object.appendChild(difficult)  # difficult标签结束

            bndbox = xmlBuilder.createElement("bndbox")  # bndbox标签
            xmin = xmlBuilder.createElement("xmin")  # xmin标签
            mathData = int(((float(oneline[1])) * Pwidth + 1) - (float(oneline[3])) * 0.5 * Pwidth)
            xminContent = xmlBuilder.createTextNode(str(mathData))
            xmin.appendChild(xminContent)
            bndbox.appendChild(xmin)  # xmin标签结束

            ymin = xmlBuilder.createElement("ymin")  # ymin标签
            mathData = int(((float(oneline[2])) * Pheight + 1) - (float(oneline[4])) * 0.5 * Pheight)
            yminContent = xmlBuilder.createTextNode(str(mathData))
            ymin.appendChild(yminContent)
            bndbox.appendChild(ymin)  # ymin标签结束

            xmax = xmlBuilder.createElement("xmax")  # xmax标签
            mathData = int(((float(oneline[1])) * Pwidth + 1) + (float(oneline[3])) * 0.5 * Pwidth)
            xmaxContent = xmlBuilder.createTextNode(str(mathData))
            xmax.appendChild(xmaxContent)
            bndbox.appendChild(xmax)  # xmax标签结束

            ymax = xmlBuilder.createElement("ymax")  # ymax标签
            mathData = int(((float(oneline[2])) * Pheight + 1) + (float(oneline[4])) * 0.5 * Pheight)
            ymaxContent = xmlBuilder.createTextNode(str(mathData))
            ymax.appendChild(ymaxContent)
            bndbox.appendChild(ymax)  # ymax标签结束

            object.appendChild(bndbox)  # bndbox标签结束

            annotation.appendChild(object)  # object标签结束

        f = open(xmlPath + name[0:-4] + ".xml", 'w')
        xmlBuilder.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='utf-8')
        f.close()

if __name__ == "__main__":
    picPath = "/mask_detection/VOCdevkit/VOC2007/JPEGImages/"  # 图片所在文件夹路径，后面的/一定要带上
    txtPath = "/mask_detection/VOCdevkit/VOC2007/YOLO/"  # txt所在文件夹路径，后面的/一定要带上
    xmlPath = "/mask_detection/VOCdevkit/VOC2007/Annotations/"  # xml文件保存路径，后面的/一定要带上
    makexml(picPath, txtPath, xmlPath)
```

**第4步：将voc格式转成yolo**

然后再转回txt的标签格式并进行训练集和测试集的划分，划分比例为8：2，代码如下：

```python
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import random
from shutil import copyfile

classes = ["0", "1", "2"]
TRAIN_RATIO = 80 # 训练集和测试集划分为8:2

def clear_hidden_files(path):
    dir_list = os.listdir(path)
    for i in dir_list:
        abspath = os.path.join(os.path.abspath(path), i)
        if os.path.isfile(abspath):
            if i.startswith("._"):
                os.remove(abspath)
        else:
            clear_hidden_files(abspath)

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(image_id):
    in_file = open('VOCdevkit/VOC2007/Annotations/%s.xml' % image_id)
    out_file = open('VOCdevkit/VOC2007/YOLOLabels/%s.txt' % image_id, 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    in_file.close()
    out_file.close()

#wd = "/mask_detection/VOCdevkit"
#wd = "/mask_detection/VOCdevkit"

wd = os.getcwd()
print(wd)
data_base_dir = os.path.join(wd, "/")
print(data_base_dir)
if not os.path.isdir(data_base_dir):
    os.mkdir(data_base_dir)
work_sapce_dir = os.path.join(data_base_dir, "VOC2007/")
if not os.path.isdir(work_sapce_dir):
    os.mkdir(work_sapce_dir)
annotation_dir = os.path.join(work_sapce_dir, "Annotations/")
if not os.path.isdir(annotation_dir):
    os.mkdir(annotation_dir)
clear_hidden_files(annotation_dir)
image_dir = os.path.join(work_sapce_dir, "JPEGImages/")
if not os.path.isdir(image_dir):
    os.mkdir(image_dir)
clear_hidden_files(image_dir)
yolo_labels_dir = os.path.join(work_sapce_dir, "YOLOLabels/")
if not os.path.isdir(yolo_labels_dir):
    os.mkdir(yolo_labels_dir)
clear_hidden_files(yolo_labels_dir)
yolov5_images_dir = os.path.join(data_base_dir, "images/")
if not os.path.isdir(yolov5_images_dir):
    os.mkdir(yolov5_images_dir)
clear_hidden_files(yolov5_images_dir)
yolov5_labels_dir = os.path.join(data_base_dir, "labels/")
if not os.path.isdir(yolov5_labels_dir):
    os.mkdir(yolov5_labels_dir)
clear_hidden_files(yolov5_labels_dir)
yolov5_images_train_dir = os.path.join(yolov5_images_dir, "train/")
if not os.path.isdir(yolov5_images_train_dir):
    os.mkdir(yolov5_images_train_dir)
clear_hidden_files(yolov5_images_train_dir)
yolov5_images_test_dir = os.path.join(yolov5_images_dir, "val/")
if not os.path.isdir(yolov5_images_test_dir):
    os.mkdir(yolov5_images_test_dir)
clear_hidden_files(yolov5_images_test_dir)
yolov5_labels_train_dir = os.path.join(yolov5_labels_dir, "train/")
if not os.path.isdir(yolov5_labels_train_dir):
    os.mkdir(yolov5_labels_train_dir)
clear_hidden_files(yolov5_labels_train_dir)
yolov5_labels_test_dir = os.path.join(yolov5_labels_dir, "val/")
if not os.path.isdir(yolov5_labels_test_dir):
    os.mkdir(yolov5_labels_test_dir)
clear_hidden_files(yolov5_labels_test_dir)

train_file = open(os.path.join(wd, "yolov5_train.txt"), 'w')
test_file = open(os.path.join(wd, "yolov5_val.txt"), 'w')
train_file.close()
test_file.close()
train_file = open(os.path.join(wd, "yolov5_train.txt"), 'a')
test_file = open(os.path.join(wd, "yolov5_val.txt"), 'a')
list_imgs = os.listdir(image_dir)  # list image files
prob = random.randint(1, 100)
print("Probability: %d" % prob)
for i in range(0, len(list_imgs)):
    path = os.path.join(image_dir, list_imgs[i])
    if os.path.isfile(path):
        image_path = image_dir + list_imgs[i]
        voc_path = list_imgs[i]
        (nameWithoutExtention, extention) = os.path.splitext(os.path.basename(image_path))
        (voc_nameWithoutExtention, voc_extention) = os.path.splitext(os.path.basename(voc_path))
        annotation_name = nameWithoutExtention + '.xml'
        annotation_path = os.path.join(annotation_dir, annotation_name)
        label_name = nameWithoutExtention + '.txt'
        label_path = os.path.join(yolo_labels_dir, label_name)
    prob = random.randint(1, 100)
    print("Probability: %d" % prob)
    if (prob < TRAIN_RATIO):  # train dataset
        if os.path.exists(annotation_path):
            train_file.write(image_path + '\n')
            convert_annotation(nameWithoutExtention)  # convert label
            copyfile(image_path, yolov5_images_train_dir + voc_path)
            copyfile(label_path, yolov5_labels_train_dir + label_name)
    else:  # test dataset
        if os.path.exists(annotation_path):
            test_file.write(image_path + '\n')
            convert_annotation(nameWithoutExtention)  # convert label
            copyfile(image_path, yolov5_images_test_dir + voc_path)
            copyfile(label_path, yolov5_labels_test_dir + label_name)
train_file.close()
test_file.close()
```

转换完之后的txt文件会更合理，如下所示：

```
2 0.5078125 0.5078125 1.0 1.0
```

最后生成的文件格式如下：

![image](https://github.com/W-cxin/mask_detection/blob/main/%E6%9C%80%E5%90%8E%E7%9A%84%E6%95%B0%E6%8D%AE%E9%9B%86%E6%A0%BC%E5%BC%8F.png)

## 2. 进行YOLOv5的训练

**第5步：准备好数据集后，进行训练**

数据集准备好后，修改数据配置文件，在data下找到voc.yaml，将其复制一份再重命名为mask.yaml，修改为如下所示：

```python
# train and val data as 1) directory: path/images/, 2) file: path/images.txt, or 3) list: [path1/images/, path2/images/]
train: /mask_detection/yolov5-5.0/VOCdevkit/images/train  # 1564 images
val: /mask_detection/yolov5-5.0/VOCdevkit/images/val  # 435 images
# number of classes
nc: 3
# class names
names: ['0','1','2']
```

还有一处要进行修改：models文件夹下的yolov5s.yaml(不同的预训练权重对应不同的网络层数，用错会报错)，同理复制yolov5.yaml为mask.yaml，打开文件进行参数修改，只需要修改类别数即可：

```python
nc: 3  # number of classes
```

运行`train.py`，结果在`/runs/train`中。

## 3. 预测和提交

**第6步：预测的label在runs中，train之后运行这一步**

运行`predict.py`，发现预测出的结果中有的图片产生了两个结果，可以改变这两个参数试试，这两个参数影响检测的阈值：

```python
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold') # 0.25
    parser.add_argument('--iou-thres', type=float, default=0.40, help='IOU threshold for NMS') # 0.45
```

**第7步：生成最后的提交文件**

```python
import os
import csv
# 要读取的文件的根目录
root_path = '/mask_detection/yolov5-5.0/runs/detect/exp3/labels/'
txtpath = '/mask_detection/yolov5-5.0/runs/detect/exp3/labels/'  # 文件夹位置
# 将所有目录下的文件信息放到列表中
def get_Write_file_infos(path):
    # 文件信息列表
    file_infos_list = []
    # 遍历并写入文件信息
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            print(filename[:-4])
            file_infos = {}
            file_infos["path"] = filename[:-4]
            f = open(txtpath + filename[:-4] + '.txt', 'r')  # 读取文本文件
            # 只取第一个检测结果
            lines = f.read()  ##Assume the sample file has 3 lines
            a = lines.split('\n', 1)[0][0]
            print(a)
            file_infos["label"] = a
            f.close()
            # 将数据追加字典到列表中
            file_infos_list.append(file_infos)
    return file_infos_list
# 写入csv文件
def write_csv(file_infos_list):
    with open('submit.csv', 'a+', newline='') as csv_file:
        headers = ['path', 'label']  # 添加第3个head元素
        csv_write = csv.DictWriter(csv_file, headers)
        csv_write.writeheader()
        #csv_writer = csv.DictWriter(csv_file, fieldnames=['path'])
        #csv_writer = csv.DictWriter(csv_file, fieldnames=['label'])
        #csv_writer.writeheader()
        for each in file_infos_list:
            csv_write.writerow(each)

def main():
    # 调用获取文件信息的函数
    file_infos_list = get_Write_file_infos(root_path)
    # 执行写入程序
    write_csv(file_infos_list)
if __name__ == '__main__':
    main()
```
