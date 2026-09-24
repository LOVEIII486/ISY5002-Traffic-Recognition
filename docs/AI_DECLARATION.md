# AI 使用声明

*中文 · [English](AI_DECLARATION.en.md)*

## 1. 范围

本项目在开发过程中使用了 AI 工具辅助。以下逐项说明所用模型、协助的环节，以及人工核验的范围。

## 2. 翻译与英文文档

英文文档由 **Gemini** 与 **GPT 5.6-luna** 协助翻译与撰写，涉及：

- `README.md`（英文版，中文原稿见 `README.zh.md`）
- `docs/ARCHITECTURE.en.md`（中文原稿见 `docs/ARCHITECTURE.md`）
- `docs/LIMITATIONS.en.md`（中文原稿见 `docs/LIMITATIONS.md`）

中文原稿由本人完成，英文版为对应译本，内容经本人核对。

## 3. 代码开发

代码的调试、解释与部分编写工作由 **Deepseek-flash** 协助，具体包括：

- 定位并解释运行时错误与异常行为；
- 解释第三方库（Ultralytics、LibCity）的接口与运行机制；
- 部分模块初稿的编写。

所有 AI 参与生成或修改的代码均由本人**严格核验**：逐段阅读、在真实数据上运行验证，并对实现方式与结果进行优化。项目的架构决策、实验设计与结果解释由本人负责。

## 4. 文献调研

使用 **Gemini** 整理并阅读了以下相关研究文献，在此过程中学习并记录了本项目所涉技术的思路，并将其应用到相应环节中：

**目标检测**

1. J. Redmon, S. Divvala, R. Girshick, A. Farhadi. "You Only Look Once: Unified, Real-Time Object Detection." *CVPR*, 2016, pp. 779–788.
   → 应用于阶段 03 的车辆检测。

**密度回归计数**

2. Y. Li, X. Zhang, D. Chen. "CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes." *CVPR*, 2018, pp. 1091–1100.
   → 应用于阶段 03b 的网络设计：全卷积结构输出密度图，密度图求和即为车辆数。

3. R. Guerrero-Gómez-Olmedo, B. Torre-Jiménez, R. López-Sastre, S. Maldonado Bascón, D. Oñoro-Rubio. "Extremely Overlapping Vehicle Counting." *IbPRIA*, 2015.
   → 即 TRANCOS_v3 数据集，用于阶段 03b 的计数网络预训练。

4. C. Zhang, H. Li, X. Wang, X. Yang. "Cross-Scene Crowd Counting via Deep Convolutional Neural Networks." *CVPR*, 2015, pp. 833–841.
   → 说明计数模型在未见场景上会显著失效、需针对目标场景微调；对应阶段 03b 的领域自适应微调。

**时空预测**

5. B. Yu, H. Yin, Z. Zhu. "Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting." *IJCAI*, 2018, pp. 3634–3640.
   → 应用于阶段 06 的 STGCN 模型。

6. J. Wang, J. Jiang, W. Jiang, C. Li, W. X. Zhao. "LibCity: An Open Library for Traffic Prediction." *SIGSPATIAL '21*, 2021, pp. 145–148.
   → 本项目阶段 05 输出的 `.geo`／`.rel`／`.dyna` 原子文件格式与阶段 06 的实验流程基于该平台。

## 5. 责任声明

本人确认以上说明属实。AI 工具在本项目中承担辅助角色；最终提交的系统、实验与结论由本人负责，其正确性已通过实际运行与人工核验予以确认。
