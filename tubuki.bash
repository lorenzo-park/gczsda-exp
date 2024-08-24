export CUDA_VISIBLE_DEVICES=0
python run.py project=tubuki gpus=1 config=timm max_epochs=50 root=/shared/lorenzo/data-tubuki-cache/exp1-cwt optimizer=adam lr=2e-3 batch_size=16 channels=3

python run.py project=tubuki gpus=1 config=dann max_epochs=50 root=/shared/lorenzo/data-tubuki-cache/exp1-cwt optimizer=sgd root_tgt=/shared/lorenzo/data-tubuki-cache/exp2-cwt lr=1e-2 batch_size=16
python run.py project=tubuki gpus=1 config=dann max_epochs=50 root=/shared/lorenzo/data-tubuki-cache/exp1-cwt optimizer=sgd root_tgt=/shared/lorenzo/data-tubuki-cache/exp3-cwt lr=1e-2 batch_size=16
python run.py project=tubuki gpus=1 config=dann max_epochs=50 root=/shared/lorenzo/data-tubuki-cache/exp1-cwt optimizer=sgd root_tgt=/shared/lorenzo/data-tubuki-cache/exp4-cwt lr=1e-2 batch_size=16
python run.py project=tubuki gpus=1 config=dann max_epochs=50 root=/shared/lorenzo/data-tubuki-cache/exp1-cwt optimizer=sgd root_tgt=/shared/lorenzo/data-tubuki-cache/exp5-cwt lr=1e-2 batch_size=16
python run.py project=tubuki gpus=1 config=dann max_epochs=50 root=/shared/lorenzo/data-tubuki-cache/exp1-cwt optimizer=sgd root_tgt=/shared/lorenzo/data-tubuki-cache/exp6-cwt lr=1e-2 batch_size=16
python run.py project=tubuki gpus=1 config=dann max_epochs=50 root=/shared/lorenzo/data-tubuki-cache/exp1-cwt optimizer=sgd root_tgt=/shared/lorenzo/data-tubuki-cache/exp7-cwt lr=1e-2 batch_size=16
python run.py project=tubuki gpus=1 config=dann max_epochs=50 root=/shared/lorenzo/data-tubuki-cache/exp1-cwt optimizer=sgd root_tgt=/shared/lorenzo/data-tubuki-cache/exp8-cwt lr=1e-2 batch_size=16
