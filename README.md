# Tubuki Pytorch Lightning

아래의 설명과 har.bash 파일을 참고.

## Training Source only model

데이터셋은 `/shared/lorenzo/data-tubuki-cache` 폴더에 spectrogram 이미지를 캐시해서 사용중.
root argument에 학습할 데이터셋 경로를 넣어주면 학습가능.
아래의 예시는 exp1 데이터셋을 cwt spectrogram으로 학습.

```python
# Source only model, root에 데이터셋 경로 지정
python run.py project=zolup-har gpus=1 config=timm max_epochs=50 root=/shared/lorenzo/data-tubuki-cache/exp1-cwt optimizer=adam lr=2e-3 batch_size=32 channels=3
```

## Training DANN models

root에 source 데이터셋 경로, root_tgt에 target 데이터셋 경로를 넣어서 source->target으로 domain adaptation하는 DANN 모델 학습. 아래의 예시는 레이블링된 exp1 cwt spectrogram 데이터셋과 레이블링이 안된 exp2 cwt spectrogram 데이터셋을 학습해서, exp2 환경에 adaptation.

```python
# Source only model, root에 데이터셋 경로 지정
python run.py project=zolup-har gpus=1 config=dann max_epochs=50 root=/shared/lorenzo/data-tubuki-cache/exp1-cwt optimizer=sgd root_tgt=/shared/lorenzo/data-tubuki-cache/exp2-cwt lr=1e-2 batch_size=32
```

## Grad-CAM 추출 스크립트

Source only 모델과 DANN domain adaptation 모델 두 모델의 grad-CAM 결과를 비교.
사용법은 `vis_grad_cam.ipynb` 파일 주석 참고.