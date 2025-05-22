# DETR Implementation

이 레포지토리는 논문  
**[DETR: End-to-End Object Detection with Transformers (Facebook AI)](https://arxiv.org/abs/2005.12872)**  
의 핵심 아이디어를 바탕으로 작성된 PyTorch 기반의 모듈형 객체 탐지 모델 구현입니다.


## 📌 개요

DETR은 **Transformer 인코더-디코더 구조**와 **이분 매칭 손실 함수(bipartite matching loss)**를 통해, 기존의 객체 탐지 방식과는 전혀 다른 새로운 접근 방식을 제안합니다.

이 프로젝트는 논문에서 제안한 구조 중 **객체 탐지에 해당하는 핵심 구성만을 구현**한 것으로, 전체 학습 파이프라인이 아닌 구조 이해 및 연구 목적으로 제작되었습니다.


## 📄 논문 리뷰

논문 리뷰 및 핵심 개념 정리는 Velog 포스트에서 확인하실 수 있습니다:  
👉 [DETR PAPER REVIEW(Velog)](https://velog.io/@davidlyoo/DETR-Paper-Review-End-to-End-Object-Detection-with-Transformers)

- 주요 개념 정리:
  - Set-based 객체 예측 방식
  - Hungarian Matching 기반 이분 매칭 손실
  - Transformer 인코더-디코더 구조
- 기존 탐지기와의 비교
- 손실 함수 설계 이유 및 구성 설명

---

## 🧠 코드 구성 및 설명

구조 분석과 확장을 용이하게 하기 위해, 각 구성요소를 모듈화하고 주요 함수에 주석을 추가하였습니다.

- `backbone/` – 특징 추출 백본 (ResNet 등)
- `transformer/` – 인코더 및 디코더 블록
- `matcher.py` – Hungarian Matching 구현
- `criterion.py` – Set-based 손실 함수 (분류 + L1 + GIoU)
- `model.py` – DETR 전체 모델 구성 래퍼

> ⚙️ 각 모듈에 상세한 주석을 통해 핵심 로직과 알고리즘 구조를 설명하였습니다.
