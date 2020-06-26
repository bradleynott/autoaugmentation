# Autoaugment Models

### Quick Links

[Changes](#changes)

[Training Details](#training-procedure)

[Training Procedure](#how-to-run)

[Prepare For Inference](#prepare-inference)

</center>

<h2 id="changes">
Train on Any Cloud Instance
</h2>

**Note:** all of the intro information outlined in the `autoaugment_tf/README.md` detailing modifications to the original source code still applies the code in this directory.

The main difference is that **_this_** approach no longer utilizes a separate SageMaker Training Job script to launch the original training script. Here, we will simply start training with the training script itself.

Additionally, you will see in the commands below that this code now works in Python 3. The main training script contains a context management modification to allow this to work.

**Note:** the training scripts in the `code` folder also differs from the SageMaker training code in that it incorporates additional data set names.

* **Additional dataset names:**

    * `cifar10_12k`: 12,000 subsample of original CIFAR-10; 10,000 train/2,000 test

    * `cifar105`: new dataset with 10,000 train/2,000 test
    
**Note:** once again these dataset names occur in `data_utils.py`, `tf_models_ps.py`, and `train_cifar_ps.py`. A change in one location will require a change in all. If you wish to modify the SageMaker code in the previous directory to accept these new model names, you will need to adjust these files accordgingly.

**Note:** `steps_per_epoch` modification necessary to improve distributed training on SageMaker has been commented out in this version of the code. This is necessary since there is no longer a SageMaker environment variable that defines number of GPUs. Instead, this version of the code is designed for a single GPU cloud instance, so the original implementation of the `steps_per_epoch` can be used.

<h2 id="training-procedure">
Training Environment
</h2>

* This code was used to train models on an IBM Cloud instance with a single Tesla V100 GPU and Ubuntu Linux 18.04 LTS Bionic Beaver Minimal Install (64 bit). To prepare the cloud instance NVIDIA GPU drivers, Docker, and nvidia-docker were installed.

Next the folder structure on the cloud instance was set up to match what the training code expetcts. Then, source data files in the `code` directory of this repo were added to the cloud isntance host machine, and necessary data files were copied onto the host machine from a Google Drive account.

Finally, a Docker container with TensorFlow was launched and used to run the training procedure.

Once model training was complete model checkpoints were downloaded to a local machine.

The entire procedure is detailed below:

## Training Procedure

<h2 id="how-to-run">
Steps to Use This Code
</h2>

1. Select a cloud provider and start an instance with a single V100 GPU and Ubuntu Linux 18.04
2. ssh into the new cloud instance
3. Install graphics drivers:

    ```
    sudo add-apt-repository ppa:graphics-drivers
    sudo apt-get update
    sudo apt-get install nvidia-driver-418
    sudo prime-select nvidia
    ```
4. Reboot instance (this will disconnect the ssh session):

    ````
    sudo reboot
    ````
    
5. Reconnect to instance once reboot is complete. Typically, a "connection refused" response under these circumstances indicates that the reboot has not completed. Try again in several minutes.
6. Verify graphics drivers installed correctly:
    ````
    nvidia-smi
    ````
**Note:** if the graphics driver install worked you should see the nvidia-smi table that provides details about your GPU

7. Install Docker:
    ````
    sudo apt-get update
    
    sudo apt-get install \
        apt-transport-https \
        ca-certificates \
        curl \
        gnupg-agent \
        software-properties-common
    
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
    
    sudo add-apt-repository \
       "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
       $(lsb_release -cs) \
       stable"
   
    sudo apt-get update
   
    sudo apt-get install docker-ce docker-ce-cli containerd.io
   
    sudo docker run hello-world
    ````
8. Install nvidia-docker2:
    ```
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
      sudo apt-key add -
  
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
 
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
      sudo tee /etc/apt/sources.list.d/nvidia-docker.list
  
    sudo apt-get update

    sudo apt-get install nvidia-docker2

    sudo pkill -SIGHUP dockerd
    ```
9. Create a `training` folder inside the cloud instance `tmp` directory. Make a `data` folder inside that directory as well:
    ```
    mkdir /tmp/training
    mkdir /tmp/training/data
    ```
10. Create directories for whichever datasets you plan to train with. For example:
    ```
    mkdir /tmp/training/data/cifar10_12k
    mkdir /tmp/training/data/cifar105
    ```
11. Change directories to the training folder and copy all of the scripts from the `code` folder in this repository
12. Download data files to the cloud instance:
    ```
    # example: download data files from a Google Drive account
    
    wget --no-check-certificate <google drive url> -O   /tmp/training/data/<dataset name>/<file name>
    ```
    **Note:** both the training and testing sets need to be placed in the same destination folder. The training script will look for both of these files when it receives the `--data_path` argument provided with the training command.
13. Start a TensorFlow Docker container and link it to the cloud instance directories for access to training scripts and data files:
    ```
    sudo docker run \
       --name=autoaug -it -v /tmp/training:/tmp -p 5000:80 \
       --runtime=nvidia tensorflow/tensorflow:1.14.0-gpu-py3 bash
    ```
14. Once inside the container use the bash prompt to navigate to the folder containing `train_cifar_ps.py`. Start training:
    ```
    python train_cifar_ps.py \
       --model_name shake_shake_96 \
       --data_path /tmp/data/cifar105 \
       --checkpoint_dir /tmp/checkpoint \
       --dataset cifar105 \
       --use_cpu 0
    ```
    **Note:** note: training procedure will create 'checkpoint' folder if it does not already exist
15. Once model training is complete, download the final checkpoints:
    ```
    # Example: using git bash to download from cloud instance to local Windows 10 machine
    
    scp -r root@<instance ip address>:/tmp/training/checkpoints/model/ "C:/Users/Brad/Downloads/ss32_cifar105"
    ```
16. When finished with checkpoint files in cloud instance and ready to train another model first clear the checkpoints folder:
    ```
    rm /tmp/checkpoints/model/*
    ```
**Note:** make sure to clear out the checkpoint folder prior to training a new model with a different architecture. The training code is designed to look for the presence of a checkpoint and start training from the last checkpoint if it finds one in the folder. If you finish training a model and then try to start training a new model with a different architecture, without first clearing out the old checkpoints, the training code will try to load an old checkpoint but it will casuse an error since the neural network architectures are different.


<h2 id="prepare-inference">
Preparing For Inference
</h2>

In order to use the checkpoints for inference, ensure that all checkpoint files are located in a folder named `model`, and then using a command prompt from within that folder, add everything to a `tar.gz` archive.

    ```
    tar -czvf model.tar.gz .
    ```
