# Simu-GAN

This project develop a Generative Adversarial Network (GAN) for High-energy physics purposes. In particular the aim of this work is to build a network that can reproduce high-energy physics events. In this case the physical process under study is a VBF process with a di-Higgs production with a semi-leptonic signature.

<p align="center">
 gg->HH->lvjj
</p>


The Monte Carlo events, used for the training, are generated at the parton level with MadGraph. 
The GAN model is implemented in Keras and Tensorflow

### Download
To download the project run the command:

```shell
>git clone https://gitlab.com/waredjeb/simu-gan.git 
```
### Requirements
To install all the required libraries run the command:

```shell 
>cd simu-gan
>pip install --user --requirement requirements.txt
```

### Run the script
Before starting the training a preprocessing on the data is needed. In this case one has to generate a map that scale the features in [-1,1]. To generate the map run the command:

```shell
>./run_scaler.sh
```
<b>NOTE</b>: The map creation is needed every time the features-space is modified.

Now the training can start. Run the command:

```shell
>./run_train.sh
```

Inside the bash script you can find the execution line: 

```shell
>python gan_train.py -e 1000000 -b 64 
```
The <b>-e</b> argument stands for the number of epochs, whilst the argument <b>-b</b> stands for the batch size

### Dataset ###
The events generated through Madgraph can be found here: https://drive.google.com/drive/folders/10VdWgzkT4O-5ySeebpAL8NwJIlqBol0c?usp=sharing
