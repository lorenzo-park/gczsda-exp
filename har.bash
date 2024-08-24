export CUDA_VISIBLE_DEVICES=0,1
# python run.py project=zolup-har gpus=1 config=timm max_epochs=50 root=/shared/lorenzo/data-tubuki-cache/exp1-cwt optimizer=adam lr=2e-3 batch_size=32 channels=3
# python run.py project=zolup-har gpus=1 config=timm max_epochs=50 root=/shared/lorenzo/data-tubuki-cache/exp1-cwt optimizer=adam lr=2e-3 batch_size=32 channels=3 seed=1
# python run.py project=zolup-har gpus=1 config=timm max_epochs=50 root=/shared/lorenzo/data-tubuki-cache/exp1-cwt optimizer=adam lr=2e-3 batch_size=32 channels=3 seed=2

# ! exp1 -> exp2
# python run.py project=zolup-har gpus=1 config=dann max_epochs=50 root=/shared/lorenzo/data-tubuki-cache/exp1-cwt optimizer=sgd root_tgt=/shared/lorenzo/data-tubuki-cache/exp2-cwt lr=1e-2 batch_size=32
# python run.py project=zolup-har gpus=1 config=dann max_epochs=50 root=/shared/lorenzo/data-tubuki-cache/exp1-cwt optimizer=sgd root_tgt=/shared/lorenzo/data-tubuki-cache/exp2-cwt lr=1e-2 batch_size=32 seed=1
# python run.py project=zolup-har gpus=1 config=dann max_epochs=50 root=/shared/lorenzo/data-tubuki-cache/exp1-cwt optimizer=sgd root_tgt=/shared/lorenzo/data-tubuki-cache/exp2-cwt lr=1e-2 batch_size=32 seed=2

# python run.py project=zolup-har gpus=1 config=adda max_epochs=50 root=/shared/lorenzo/data-tubuki-cache/exp1-cwt optimizer=adam root_tgt=/shared/lorenzo/data-tubuki-cache/exp2-cwt lr=5e-5 batch_size=32 pretrained="/root/dezsda/checkpoints/timm-mobilenet_v2-epoch\=11-task\=nexmon-val_loss\=0.6735.ckpt"
# python run.py project=zolup-har gpus=1 config=adda max_epochs=50 root=/shared/lorenzo/data-tubuki-cache/exp1-cwt optimizer=adam root_tgt=/shared/lorenzo/data-tubuki-cache/exp2-cwt lr=5e-5 batch_size=32 pretrained="/root/dezsda/checkpoints/timm-mobilenet_v2-epoch\=11-task\=nexmon-val_loss\=0.6735.ckpt" seed=1
# python run.py project=zolup-har gpus=1 config=adda max_epochs=50 root=/shared/lorenzo/data-tubuki-cache/exp1-cwt optimizer=adam root_tgt=/shared/lorenzo/data-tubuki-cache/exp2-cwt lr=5e-5 batch_size=32 pretrained="/root/dezsda/checkpoints/timm-mobilenet_v2-epoch\=11-task\=nexmon-val_loss\=0.6735.ckpt" seed=2

# python run.py project=zolup-har gpus=2 config=cycada max_epochs=50 root=/shared/lorenzo/data-tubuki-cache/exp1-cwt optimizer=adam root_tgt=/shared/lorenzo/data-tubuki-cache/exp2-cwt lr=1e-4 batch_size=12 pretrained="/root/dezsda/checkpoints/timm-mobilenet_v2-epoch\=11-task\=nexmon-val_loss\=0.6735.ckpt" grad_accum=4 channels=3 num_blocks=9 lambda_idt=1.0
# python run.py project=zolup-har gpus=2 config=gcada max_epochs=50 root=/shared/lorenzo/data-tubuki-cache/exp1-cwt optimizer=adam root_tgt=/shared/lorenzo/data-tubuki-cache/exp2-cwt lr=1e-4 batch_size=16 pretrained="/root/dezsda/checkpoints/timm-mobilenet_v2-epoch\=11-task\=nexmon-val_loss\=0.6735.ckpt" grad_accum=4 channels=3 num_blocks=9 transformation=rotate lambda_idt=1.0

# ! exp1 -> exp4
# python run.py project=zolup-har gpus=1 config=dann max_epochs=50 root=/shared/lorenzo/data-tubuki-cache/exp1-cwt optimizer=sgd root_tgt=/shared/lorenzo/data-tubuki-cache/exp4-cwt lr=1e-2 batch_size=32
# python run.py project=zolup-har gpus=1 config=dann max_epochs=50 root=/shared/lorenzo/data-tubuki-cache/exp1-cwt optimizer=sgd root_tgt=/shared/lorenzo/data-tubuki-cache/exp4-cwt lr=1e-2 batch_size=32 seed=1
# python run.py project=zolup-har gpus=1 config=dann max_epochs=50 root=/shared/lorenzo/data-tubuki-cache/exp1-cwt optimizer=sgd root_tgt=/shared/lorenzo/data-tubuki-cache/exp4-cwt lr=1e-2 batch_size=32 seed=2

# python run.py project=zolup-har gpus=1 config=adda max_epochs=50 root=/shared/lorenzo/data-tubuki-cache/exp1-cwt optimizer=adam root_tgt=/shared/lorenzo/data-tubuki-cache/exp4-cwt lr=5e-5 batch_size=32 pretrained="/root/dezsda/checkpoints/timm-mobilenet_v2-epoch\=11-task\=nexmon-val_loss\=0.6735.ckpt"
# python run.py project=zolup-har gpus=1 config=adda max_epochs=50 root=/shared/lorenzo/data-tubuki-cache/exp1-cwt optimizer=adam root_tgt=/shared/lorenzo/data-tubuki-cache/exp4-cwt lr=5e-5 batch_size=32 pretrained="/root/dezsda/checkpoints/timm-mobilenet_v2-epoch\=11-task\=nexmon-val_loss\=0.6735.ckpt" seed=1
# python run.py project=zolup-har gpus=1 config=adda max_epochs=50 root=/shared/lorenzo/data-tubuki-cache/exp1-cwt optimizer=adam root_tgt=/shared/lorenzo/data-tubuki-cache/exp4-cwt lr=5e-5 batch_size=32 pretrained="/root/dezsda/checkpoints/timm-mobilenet_v2-epoch\=11-task\=nexmon-val_loss\=0.6735.ckpt" seed=2

# python run.py project=zolup-har gpus=2 config=cycada max_epochs=50 root=/shared/lorenzo/data-tubuki-cache/exp1-cwt optimizer=adam root_tgt=/shared/lorenzo/data-tubuki-cache/exp4-cwt lr=1e-4 batch_size=12 pretrained="/root/dezsda/checkpoints/timm-mobilenet_v2-epoch\=11-task\=nexmon-val_loss\=0.6735.ckpt" grad_accum=4 channels=3 num_blocks=9 lambda_idt=1.0
# python run.py project=zolup-har gpus=2 config=gcada max_epochs=50 root=/shared/lorenzo/data-tubuki-cache/exp1-cwt optimizer=adam root_tgt=/shared/lorenzo/data-tubuki-cache/exp4-cwt lr=1e-4 batch_size=16 pretrained="/root/dezsda/checkpoints/timm-mobilenet_v2-epoch\=11-task\=nexmon-val_loss\=0.6735.ckpt" grad_accum=4 channels=3 num_blocks=9 transformation=rotate lambda_idt=1.0

# ! exp1 -> exp6
# python run.py project=zolup-har gpus=1 config=dann max_epochs=50 root=/shared/lorenzo/data-tubuki-cache/exp1-cwt optimizer=sgd root_tgt=/shared/lorenzo/data-tubuki-cache/exp6-cwt lr=1e-2 batch_size=32
# python run.py project=zolup-har gpus=1 config=dann max_epochs=50 root=/shared/lorenzo/data-tubuki-cache/exp1-cwt optimizer=sgd root_tgt=/shared/lorenzo/data-tubuki-cache/exp6-cwt lr=1e-2 batch_size=32 seed=1
# python run.py project=zolup-har gpus=1 config=dann max_epochs=50 root=/shared/lorenzo/data-tubuki-cache/exp1-cwt optimizer=sgd root_tgt=/shared/lorenzo/data-tubuki-cache/exp6-cwt lr=1e-2 batch_size=32 seed=2

# python run.py project=zolup-har gpus=1 config=adda max_epochs=50 root=/shared/lorenzo/data-tubuki-cache/exp1-cwt optimizer=adam root_tgt=/shared/lorenzo/data-tubuki-cache/exp6-cwt lr=5e-5 batch_size=32 pretrained="/root/dezsda/checkpoints/timm-mobilenet_v2-epoch\=11-task\=nexmon-val_loss\=0.6735.ckpt"
# python run.py project=zolup-har gpus=1 config=adda max_epochs=50 root=/shared/lorenzo/data-tubuki-cache/exp1-cwt optimizer=adam root_tgt=/shared/lorenzo/data-tubuki-cache/exp6-cwt lr=5e-5 batch_size=32 pretrained="/root/dezsda/checkpoints/timm-mobilenet_v2-epoch\=11-task\=nexmon-val_loss\=0.6735.ckpt" seed=1
# python run.py project=zolup-har gpus=1 config=adda max_epochs=50 root=/shared/lorenzo/data-tubuki-cache/exp1-cwt optimizer=adam root_tgt=/shared/lorenzo/data-tubuki-cache/exp6-cwt lr=5e-5 batch_size=32 pretrained="/root/dezsda/checkpoints/timm-mobilenet_v2-epoch\=11-task\=nexmon-val_loss\=0.6735.ckpt" seed=2

# python run.py project=zolup-har gpus=2 config=cycada max_epochs=50 root=/shared/lorenzo/data-tubuki-cache/exp1-cwt optimizer=adam root_tgt=/shared/lorenzo/data-tubuki-cache/exp6-cwt lr=1e-4 batch_size=12 pretrained="/root/dezsda/checkpoints/timm-mobilenet_v2-epoch\=11-task\=nexmon-val_loss\=0.6735.ckpt" grad_accum=4 channels=3 num_blocks=9 lambda_idt=1.0
# python run.py project=zolup-har gpus=2 config=gcada max_epochs=50 root=/shared/lorenzo/data-tubuki-cache/exp1-cwt optimizer=adam root_tgt=/shared/lorenzo/data-tubuki-cache/exp6-cwt lr=1e-4 batch_size=16 pretrained="/root/dezsda/checkpoints/timm-mobilenet_v2-epoch\=11-task\=nexmon-val_loss\=0.6735.ckpt" grad_accum=4 channels=3 num_blocks=9 transformation=rotate lambda_idt=1.0
