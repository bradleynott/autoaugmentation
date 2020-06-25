# ref: https://www.youtube.com/watch?v=x94hpOmKtXM
# ref: https://github.com/aws-samples/amazon-sagemaker-script-mode/blob/master/tf-distribution-options/tf-distributed-training.ipynb
# ref: https://aws.amazon.com/blogs/machine-learning/launching-tensorflow-distributed-training-easily-with-horovod-or-parameter-servers-in-amazon-sagemaker/
# ref: https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/estimator.py

import numpy as np
import os
import sagemaker
from sagemaker import get_execution_role

sagemaker_session = sagemaker.Session()
role = get_execution_role()

bucket = sagemaker_session.default_bucket()
print('Bucket:\n{}'.format(bucket))

#inputs = 's3://sagemaker-us-east-2-955307202111/data/cifar10'
#inputs = 's3://sagemaker-us-east-2-955307202111/data/cifar102'
#inputs = 's3://sagemaker-us-east-2-955307202111/data/cifar10_10k'
#inputs = 's3://sagemaker-us-east-2-955307202111/data/cifar10_30k'
inputs = 's3://sagemaker-us-east-2-955307202111/data/cifar102_30k'

from sagemaker.tensorflow import TensorFlow

ps_instance_type = 'ml.p3.2xlarge' # 1xV100
ps_instance_count = 1

model_dir = "/opt/ml/model"

distributions = {'parameter_server': {
                    'enabled': True}
                }

# Note: num_epochs is assigned in a conditional statement within train_cifar_ps.py
#       main block; it must be manually edited in that file, not adjusted here

# these hyperparameter names correspond to the command line arguments
# in the training script parsed in the main function
hyperparameters = {'model_name': 'wrn',
                   #'num_epochs': 1, # comment out for full run; param is included if 'wrn' is given
                   #'checkpoint_dir': '/tmp/wrn_training',
                   'checkpoint_dir': '/opt/ml/model',
                   #'data_path': '/tmp/data',
                   'dataset': 'cifar102_30k'}
                   #'use_cpu': 0}

estimator_ps = TensorFlow( base_job_name='wrn-cifar102-30k-single',
                           train_max_run=48 * 60 * 60,
                           source_dir='/home/ec2-user/SageMaker',
                           entry_point='train_cifar_ps.py',
                           role=role,
                           framework_version='1.13',
                           py_version='py2',
                           script_mode=True,
                           hyperparameters=hyperparameters,
                           train_instance_count=ps_instance_count,
                           #train_instance_count=1,
                           train_instance_type=ps_instance_type, 
                           #train_instance_type='local_gpu',
                           model_dir=model_dir,
                           distributions=distributions )
                           
# key: command line argument
# value: s3 location of file
#remote_inputs = {'data_path': inputs} # THIS DEFINES AN SM CHANNEL 'data_path'

remote_inputs = {'batches' : inputs+'/training'}

estimator_ps.fit(remote_inputs, wait=True)
