export CUDA_VISIBLE_DEVICES=0,1,2
python run.py gpus=1 config=timm task=digit backbone_name=lenet max_epochs=50 root=/root/dezsda/data/FashionMNIST_G lr=1e-3 optimizer=adam batch_size=64
python run.py gpus=1 config=timm task=digit backbone_name=lenet max_epochs=50 root=/root/dezsda/data/FashionMNIST_G lr=1e-3 optimizer=adam batch_size=64 seed=1
python run.py gpus=1 config=timm task=digit backbone_name=lenet max_epochs=50 root=/root/dezsda/data/FashionMNIST_G lr=1e-3 optimizer=adam batch_size=64 seed=2
python run.py gpus=1 config=timm task=digit backbone_name=lenet max_epochs=50 root=/root/dezsda/data/FashionMNIST_G lr=1e-3 optimizer=adam batch_size=64 seed=3
