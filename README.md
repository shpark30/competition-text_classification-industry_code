[![Colab](https://img.shields.io/static/v1?label=View&message=PDF&color=red)](./competition-text_classification-industry_code/2022_통계청_AI활용대회_리뷰.pdf)

상단 PDF에서 자세한 리뷰를 확인할 수 있습니다.

## Description
**Competition Link : [통계데이터 인공지능 활용대회]([https://data.kostat.go.kr/sbchome/bbs/boardDetail.do])<p>**

**[주제]**

2022년도 통계청이 주관하는 AI활용대회로 산업을 설명하는 텍스트 데이터로 산업분류코드를 분류

**[전략]**

다양한 거대 언어모델을 Bagging

**[평가 방식]**

정확도(Accuracy) + 정성 평가

**[최종 점수]**

- Accuracy : 91.03
- F1-score : 77.84

**[결과]**
 
- 1차 : Pass
- 2차 : Fail 

정확도는 수상권이었지만 탈락했던 이유는 거대 언어 모델을 앙상블하는 프로세스가 너무 무겁기 때문이라고 생각합니다.

<br>

---

<br>

**[대상 모델]**

1. kobert
2. mlbert
3. bert
4. albert
5. kobart
6. asbart
7. kogpt2
8. kogpt3
9. electra
10. funnel

<br>

---

<br>

## 주요 모듈
- train.py : 단일 모델 학습
- bagging.py : 10종의 언어모델을 조합해 앙상블(bagging) 학습
- inference.py : 단일 모델을 이용해 inference
- load.py : pretrained 언어모델의 tokenizer와 model weights를 불러오는 모듈
- dataset.py : 언어모델별로 산업설명 텍스트 데이터를 tokenize하는 모듈
- network.py : 각 언어모델을 base로 하는 텍스트 분류 모델을 정의
- utils.py
- notebook/ensemble_test_inferece.ipynb : bagging.py로 학습한 base model들로 inference 후 앙상블 결과를 도출
- command/train.sh : train.py를 실행하는 shell script
- command/bagging.sh : bagging.py를 실행하는 shell script
- command/inference.sh : inference.py를 실행하는 shell script
