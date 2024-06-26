# Recalling MultiView History to Future: Cognition Based Temporal Knowledge Graph Reasoning

This is the released codes of the following paper submitted to ICDE 2025:

Kangzheng Liu, Feng Zhao, Guandong Xu, Huan Huo. Recalling MultiView History to Future: Cognition Based Temporal Knowledge Graph Reasoning.

![RMVH](https://github.com/Liudaxian1/FIG/blob/main/RMVH.png)

## Environment

```shell
python==3.10.9
torch==2.2.1+cu118
dgl==2.1.0+cu118
tqdm==4.66.2
numpy==1.26.4
```

## Introduction

- ``src``: Python scripts.
- ``results``: Model files that replicate the reported results in out paper.
- ``data``: TKGs used in the experiments.

## Training Command

```shell
python main.py --model RMVH --dataset ICEWS14 --bias learn --s-delta-ind --n-head 2 --rank 20 --history-len 6
```

```shell
python main.py --model RMVH --dataset ICEWS05-15 --bias learn --s-delta-ind --n-head 2 --rank 20 --history-len 9
```

```shell
python main.py --model RMVH --dataset ICEWS18 --bias learn --s-delta-ind --n-head 2 --rank 20 --history-len 12
```

```shell
python main.py --model RMVH --dataset GDELT --bias learn --s-delta-ind --n-head 2 --rank 20 --history-len 9
```

## Testing Command

```shell
python main.py --model RMVH --dataset ICEWS14 --bias learn --s-delta-ind --n-head 2 --rank 20 --history-len 6 --test
```

```shell
python main.py --model RMVH --dataset ICEWS05-15 --bias learn --s-delta-ind --n-head 2 --rank 20 --history-len 9 --test
```

```shell
python main.py --model RMVH --dataset ICEWS18 --bias learn --s-delta-ind --n-head 2 --rank 20 --history-len 12 --test
```

```shell
python main.py --model RMVH --dataset GDELT --bias learn --s-delta-ind --n-head 2 --rank 20 --history-len 9 --test
```
