# AIHUB 한국인 재식별 이미지 모델
AIHUB의 한국인 재식별 이미지 데이터를 학습한 ReID 모델입니다.   
   
   
# Reference
데이터셋 링크 : https://aihub.or.kr/aidata/7977   
학습에 사용된 모델 : https://github.com/NVlabs/SegFormer   
학습에 사용된 loss :  https://arxiv.org/pdf/1812.02465.pdf   


# Installation
<pre>
<code>
pip install torch
pip install scikit-learn
pip install opencv-python
pip install albumentations
pip install timm
</code>
</pre>

# Train
<pre>
<code>
python train.py --config cfg.py
</code>
</pre>

# Result
- AIHUB validation   
AIHUB 데이터셋 문서에는 query/test로 나누어서 평가를 진행하였다고 쓰여있으나, 해당 리스트가 공개 되어있지 않음.   
따라서 validation set 전체를 query+test셋으로 평가 진행   

|Dataset|Model|Rank1 Acc|mAP|
|---|---|---|---|
|AIHUB 한국인 재식별 Training set|MixFormer-b0|0.9997|0.9879|

- Market 1501 test protocol   
   
|Dataset|Model|Rank1 Acc|mAP|
|---|---|---|---|
|AIHUB 한국인 재식별 Training set|MixFormer-b0|0.9225|0.9593|

# Evaluate
AIHUB validation 데이터 평가   
<pre>
<code>
python evaluate.py --config cfg.py --dataset AIHUB
</code>
</pre>

Market1501 Test protocol 평가
<pre><code>
python evaluate.py --config cfg.py --dataset Market1501 --dir_query [Market1501 query 폴더 경로] --dir_test [Market1501 bounding_box_test 폴더 경로]
</code></pre>

# Pretrained Model   

|Dataset|Backbone|Link|
|---|---|---|
|AIHUB 한국인 재식별 Training set|MixFormer b0|[Download](https://drive.google.com/file/d/1c2qhJBh4-kMpdB6v31c2bt56SAiK1DMc/view?usp=sharing)|
