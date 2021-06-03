# Adversarial Domain Adaptation for Sensor Networks

## What is this
This software was made over the course of my master dissertation at INESCTEC-Porto. In it, I studied how to implement domain adaptation models that considered the temporal component of data.

In total, I proposed 4 new domain adaptation models, that can be run with the code in this repository: Simple Model, SingleLSTM Model, DoubleLSTM Model and CommonLSTM Model. 

These models were inspired by the FCN-rLSTM network, described in paper:

`Zhang et al., "FCN-rLSTM: Deep spatio-temporal neural networks for vehicle counting in city cameras", ICCV 2017.`


and by the algorithm MDAN, described in paper:

`Zhao et al., "Adversarial Multiple Source Domain Adaptation", Advances in Neural Information Processing Systems 2018.`

### FCN-rLSTM

![alt text](https://imgur.com/aly17za.png)

This is the original model described in paper by Zhang et al. It can be run with commands:

`python3 src/run.py --model=original`, in case the temporal component of data should not be considered.

`python3 src/run.py --model=original_temporal`, in case the temporal component of data should be considered.

### Simple Model

![alt text](https://imgur.com/HPdMQGe.png)

It can be run with command:

`python3 src/run.py --model=simple`

### SingleLSTM

![alt text](https://imgur.com/ZYQZmR5.png)

It can be run with command:

`python3 src/run.py --model=single`

### DoubleLSTM

![alt text](https://imgur.com/ZIEcDAO.png)

It can be run with command:

`python3 src/run.py --model=double`

### CommonLSTM

![alt text](https://imgur.com/oKHl1EW.png)

It can be run with command:

`python3 src/run.py --model=common`

