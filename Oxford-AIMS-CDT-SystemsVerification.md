# Oxford AIMS CDT Systems Verification (HT, 2020)

## Lab Session: DeepGame

This is the webpage for the lab session of the AIMS CDT __Systems Verification__ course. 

To better understand the tool, feel free to look into the accompanying published paper: 
[A game-based approximate verification of deep neural networks with provable guarantees](https://www.sciencedirect.com/science/article/pii/S0304397519304426).

### 0. Installation of DeepGame

Download the __DeepGame__ tool from https://github.com/minwu-cs/DeepGame. In order to run the tool, _Python_ and some other packages such as _keras_ and _numpy_ are needed. 

Below is a list of the developer's platform for reference in case there is package inconsistency.

###### Developer's Platform
```
python 3.5.5
keras 2.1.3
tensorflow-gpu 1.4.0
numpy 1.14.3
matplotlib 2.2.2
scipy 1.1.0
```

Use the following command line to run DeepGame. 

###### Run
```
python main.py mnist ub cooperative 67 L2 10 1
```

Detailed explaination of each parameter is as follows.
- `main.py` is the main Python file. 
- `mnist` refers to the name of the MNIST dataset. It can be other datasets, such as `cifar10` and `gtsrb`.
- `ub` denotes upper bound. Use `lb` for lower bound.
- `cooperative` indicates the computation of the _maximum safe radius_ in a cooperative game. To compute the _feature robustness_, use `competitive` instead.
- `67` is the index of the image in the dataset.
- `L2` is short for the _L<sup>2</sup> norm_, i.e., the Euclidean distance. Use `L1` for the _L<sup>1</sup> norm_ (Manhattan distance), or `L0` for the Hamming distance.
- `10` gives the distance budget.
- `1` denotes the value of _atomic manipulation_ imposed on each pixel/channel of an image.

Alternatively, you may run DeepGame via the following command line, which allows a sequential execution of the above command line.
```
./commands.sh
```
-------------------

### 1. Adversarial Examples

### 2. Robustness of Deep Neural Networks

![alt text](figures/Cooperative_MNIST.png)

-------------------

### 3. Generation of Saliency Maps

![alt text](figures/Feature.png)



### Citation
