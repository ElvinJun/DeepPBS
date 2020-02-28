## DeepPBS
Motivation:Accurate all-atom protein structures play an important role in various research and applications. However, in most cases, only coarse-grained models can be obtained for reasons. Precisely predict protein backbone structures based on alpha-carbon traces, the most-used coarse-grained model, is a pivotal step for precise all-atom modeling for protein structures. 

Results: In this study, we proposed a deep learning-based method to predict protein backbone structures from alpha-carbon traces. Our method achieved comparable performance as the best previous method with cRMSD between predicted coordinates and reference coordinates as measurement.

[点击进入骨架结构预测网页](deeppbs.com)

[image](https://github.com/ElvinJun/DeepPBS/edit/master/process.jpg)
# Recurrent Geometric Networks
This is the reference (TensorFlow) implementation of recurrent geometric networks (RGNs), described in the paper [End-to-end differentiable learning of protein structure](https://www.biorxiv.org/content/early/2018/08/29/265231). 

## Installation and requirements
Extract all files in the [model](https://github.com/aqlaboratory/rgn/tree/master/model) directory in a single location and use `protling.py`, described further below, to train new models and predict structures. Below are the language requirements and package dependencies:

* Python 2.7
* TensorFlow >= 1.4 (tested up to 1.12)
* setproctitle

## Usage
The [`protling.py`](https://github.com/aqlaboratory/rgn/blob/master/model/protling.py) script facilities training of and prediction using RGN models. Below are typical use cases. The script also accepts a number of command-line options whose functionality can be queried using the `--help` option.

#### Train a new model or continue training an existing model
RGN models are described using a configuration file that controls hyperparameters and architectural choices. For a list of available options and their descriptions, see its [documentation](https://github.com/aqlaboratory/rgn/blob/master/CONFIG.md). Once a configuration file has been created, along with a suitable dataset (download a ready-made [ProteinNet](https://github.com/aqlaboratory/proteinnet) data set or create a new one from scratch using the [`convert_to_tfrecord.py`](https://github.com/aqlaboratory/rgn/blob/master/model/convert_to_tfrecord.py) script), the following directory structure must be created:

```
<baseDirectory>/runs/<runName>/<datasetName>/<configurationFile>
<baseDirectory>/data/<datasetName>/[training,validation,testing]
```

Where the first path points to the configuration file and the second path to the directories containing the training, validation, and possibly test sets. Note that `<runName>` and `<datasetName>` are user-defined variables specified in the configuration file that encode the name of the model and dataset, respectively.

Training of a new model can then be invoked by calling:

```
python protling.py [configurationFilePath] -d [baseDirectory]
```

Download a pre-trained model for an example of a correctly defined directory structure. Note that ProteinNet training sets come in multiple "thinnings" and only one should be used at a time by placing it in the main training directory.

To resume training an existing model, run the command above for a previously trained model with saved checkpoints.

#### Predict new structures using a trained model
To predict the structure of a new protein using an existing model with a saved checkpoint, call:

```
python protling.py [configFilePath] -d [baseDirectory] -p
```



