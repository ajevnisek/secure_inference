# Docker instructions
1. Build the docker:
```shell
docker build -t mmclassification docker/
```
2. Then run the docker:
On the A5000:
```shell
docker run --gpus all -v $(pwd):/home/code -v $(readlink -f trained_networks):/home/code/trained_networks -it mmclassification:latest  /bin/bash
```
On the 1080:
```shell
docker run --gpus all -v $(pwd):/home/code -v $(readlink -f trained_networks):/home/code/trained_networks -it mmclassification:latest  /bin/bash
```

3. Then reinstall mmpretrain locally (I was too lazy to do it with the docker):
```shell
pip install -e .
pip install scikit-learn
```
4. Hit folder back to go to the main.
```shell
cd ..
```
