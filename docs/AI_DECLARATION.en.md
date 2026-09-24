# AI Usage Declaration

*English · [中文](AI_DECLARATION.md)*

## 1. Scope

AI tools were used in the development of this project. This document states which models were used, which parts of the work they assisted with, and the extent of human verification.

## 2. Translation and English documentation

The English documentation was translated and drafted with the assistance of **Gemini** and **GPT 5.6-luna**, covering:

- `README.md` (English edition; the Chinese original is `README.zh.md`)
- `docs/ARCHITECTURE.en.md` (Chinese original: `docs/ARCHITECTURE.md`)
- `docs/LIMITATIONS.en.md` (Chinese original: `docs/LIMITATIONS.md`)

The Chinese originals were written by the author; the English editions are corresponding translations, and their content was checked by the author.

## 3. Code development

Debugging, explanation and part of the implementation were assisted by **Deepseek-flash**, specifically:

- locating and explaining runtime errors and unexpected behaviour;
- explaining the interfaces and behaviour of third-party libraries (Ultralytics, LibCity);
- drafting parts of some modules.

All code that AI contributed to was **strictly verified** by the author: read through section by section, executed against real data, and then optimised in both implementation and outcome. The architectural decisions, experimental design and interpretation of results are the author's own.

## 4. Literature review

**Gemini** was used to gather and read the following related research literature, learning and recording the ideas behind the techniques this project relies on and applying them to the corresponding stages:

**Object detection**

1. J. Redmon, S. Divvala, R. Girshick, A. Farhadi. "You Only Look Once: Unified, Real-Time Object Detection." *CVPR*, 2016, pp. 779–788.
   → Applied to vehicle detection in stage 03.

**Density-regression counting**

2. Y. Li, X. Zhang, D. Chen. "CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes." *CVPR*, 2018, pp. 1091–1100.
   → Applied to the network design in stage 03b: a fully-convolutional structure emitting a density map, whose sum is the vehicle count.

3. R. Guerrero-Gómez-Olmedo, B. Torre-Jiménez, R. López-Sastre, S. Maldonado Bascón, D. Oñoro-Rubio. "Extremely Overlapping Vehicle Counting." *IbPRIA*, 2015.
   → The TRANCOS_v3 dataset, used to pretrain the counting network in stage 03b.

4. C. Zhang, H. Li, X. Wang, X. Yang. "Cross-Scene Crowd Counting via Deep Convolutional Neural Networks." *CVPR*, 2015, pp. 833–841.
   → Shows that counting models degrade markedly on unseen scenes and must be fine-tuned for the target scene; corresponds to the domain-adaptive fine-tuning in stage 03b.

**Spatio-temporal forecasting**

5. B. Yu, H. Yin, Z. Zhu. "Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting." *IJCAI*, 2018, pp. 3634–3640.
   → Applied to the STGCN model in stage 06.

6. J. Wang, J. Jiang, W. Jiang, C. Li, W. X. Zhao. "LibCity: An Open Library for Traffic Prediction." *SIGSPATIAL '21*, 2021, pp. 145–148.
   → The `.geo` / `.rel` / `.dyna` atomic file format emitted by stage 05, and the experiment workflow of stage 06, are based on this platform.

## 5. Statement of responsibility

The author confirms the above to be accurate. AI tools played a supporting role in this project; the system, experiments and conclusions submitted are the author's responsibility, and their correctness has been confirmed through actual execution and human review.
