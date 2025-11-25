# ViZDoom Setup Guide - TCC

This guide explains how to install and configure ViZDoom on Linux (Ubuntu distributions), including environment preparation, dependencies, Python virtual environment setup, WAD configuration, and a basic test script.

I used Linux/mint to do this work, to work on windows you will need some other way, see *Some Links* below.

## Prerequisites

Make sure your system is up to date:

```
sudo apt update && sudo apt upgrade -y

```

Required packages:

* Python 3.8+

* pip

* git

* cmake

* g++

Install everything with:
```
sudo apt install python3 python3-pip python3-venv git cmake g++ -y
```

## Install Required Dependencies

ViZDoom depends on several system libraries:

sudo apt install libboost-all-dev libsdl2-dev zlib1g-dev libjpeg-dev -y

##Create a Python Virtual Environment (Recommended)
python3 -m venv vizdoom-env


## Activate the environment:

```
source vizdoom-env/bin/activate
```
## Get all Dependencies
```
pip install -r requirements.txt
```


## Install TensorBoard Support
```
pip install tensorboard
```

Run TensorBoard:
```
tensorboard --logdir logs/
```

## Deactivate the Virtual Environment
```
deactivate
```

### Some Links
* https://vizdoom.cs.put.edu.pl/
* https://vizdoom.farama.org/
