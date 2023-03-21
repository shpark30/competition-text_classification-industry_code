# industry_code_classify
## 자세한 리뷰는 "2022_통계청_AI활용대회_리뷰.pdf" 파일을 참고해주세요.

- 2022년도 통계청이 주관하는 AI활용대회로 산업을 설명하는 텍스트 데이터로 산업분류코드를 분류하는 과제입니다.
- 다양한 거대 언어모델을 Bagging하는 시도를 했습니다.
- 정확도와 정성 평가를 진행합니다. 정확도는 수상권이었지만 탈락했던 이유로 너무 많은 모델을 앙상블하는 무거운 프로세스를 채택했기 때문이라고 추측합니다.

## 설명
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
