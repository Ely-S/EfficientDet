
## Working with preemptible TPUs

Create the instance

    ctpu up -tf-version 1.15 -preemptible -tpu-only

When the TPU gets preempted run:

    ctpu delete -tpu-only                
    ctpu up -tf-version 1.15 -preemptible -tpu-only


## Install

run the installs from the dockerfile

    sudo apt-get update
    sudo apt-get install -y git libsm6 libxext6 libxrender1 python3-dev

Note that specifying the python and pip versions is important.

    sudo pip3 install Cython
    sudo pip3 install -r requirements_tpu.txt
    python3 setup.py build_ext --inplace


## Generating Data

    mkdir -p run1/train run1/val
 
    python3 make_tpu_dataset.py --phi 0 --split validation --path run1/val --debug --num-shards 20

   python3 make_tpu_dataset.py --phi 0 --split validation --path run1/train --debug --num-shards 20

    gsutil -m cp -r run1  gs://ondaka-ml-data/dev/run1
