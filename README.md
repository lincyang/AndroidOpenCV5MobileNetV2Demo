# OpenCV Image Classifier

基于 **OpenCV 5.0 DNN** 的 Android 实时图像分类示例。

使用摄像头采集画面，通过 MobileNet V2 对 ImageNet 1000 类进行实时分类，并在预览画面与底部面板展示 Top-3 结果及推理耗时。

**作者：** 程序员Linc  
**公众号：** 程序员Linc

---

## 功能特性

- OpenCV 5.0 官方 Maven 依赖（`org.opencv:opencv:5.0.0`）
- `JavaCamera2View` 相机预览 + `Dnn.readNetFromONNX` 模型加载
- MobileNet V2 fp32 / int8 双模型切换
- 画面叠加 Top-3 分类标签与推理耗时

## 环境要求

- Android Studio 2022.1+
- Android SDK 24+
- 带摄像头的 Android 真机

## 快速开始

### 1. 下载模型

```bash
cd image_classification
python -m pip install -r requirements.txt
python prepare_models.py --output_dir ./app/src/main/res/raw
```

将下载以下文件到 `app/src/main/res/raw/`：

| 文件 | 说明 |
|------|------|
| `mobilenetv2_fp32.onnx` | FP32 模型（推荐） |
| `mobilenetv2_int8.onnx` | INT8 量化模型 |
| `imagenet_classes.txt` | ImageNet 类别标签 |

### 2. 编译运行

用 Android Studio 打开本项目根目录，连接真机，点击 **Run** 即可。

## 项目结构

```
app/src/main/java/com/programmerlinc/opencv/imageclassifier/
├── MainActivity.kt          # 相机预览与推理入口
├── OpenCVModelLoader.kt     # 从 res/raw 加载 DNN 模型
├── OpenCVClassifier.kt      # 预处理、推理、结果绘制
├── ClassificationUtils.kt   # Softmax 与 Top-K
└── ImageUtil.kt             # 输入尺寸常量
```

## 技术说明

- 模型来源：[ONNX Model Zoo - MobileNet V2](https://github.com/onnx/models/tree/main/vision/classification/mobilenet)
- 预处理：ImageNet 标准归一化（mean / std），224×224 输入
- 推理引擎：OpenCV DNN（`setInput` → `forward` → `softmax`）

## License

[MIT License](LICENSE)

## 联系方式

- GitHub: https://github.com/lincyang
- 微信公众号: 程序员Linc

欢迎关注公众号获取更多技术文章和项目更新！