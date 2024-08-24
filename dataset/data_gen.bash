set -ex
# python data_gen.py dataset=MNIST domain=C
python data_gen.py dataset=MNIST domain=E
python data_gen.py dataset=MNIST domain=N

python data_gen.py dataset=EMNIST domain=C
python data_gen.py dataset=EMNIST domain=E
python data_gen.py dataset=EMNIST domain=N

python data_gen.py dataset=FashionMNIST domain=C
python data_gen.py dataset=FashionMNIST domain=E
python data_gen.py dataset=FashionMNIST domain=N