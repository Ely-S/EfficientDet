
## Working with preemptible TPUs

Create the instance

    ctpu up -tf-version 1.15 -preemptible -tpu-only

When the TPU gets preempted run:

    ctpu delete -tpu-only                
    ctpu up -tf-version 1.15 -preemptible -tpu-only

## Run on GPU instance


        gcloud beta compute \
            --project=ondaka-recognition-demo instances create gpu3 \
            --zone=us-west1-b \
            --machine-type=n1-standard-8 \
            --subnet=default \
            --network-tier=PREMIUM \
            --no-restart-on-failure \
            --maintenance-policy=TERMINATE \
            --preemptible \
            --service-account=608926314807-compute@developer.gserviceaccount.com \
            --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append \
            --accelerator=type=nvidia-tesla-v100,count=8 \
            --image=tf2-latest-gpu-20200130 \
            --image-project=deeplearning-platform-release \
            --boot-disk-size=200GB \
            --no-boot-disk-auto-delete \
            --boot-disk-type=pd-ssd \
            --boot-disk-device-name=gpu3 \
            --reservation-affinity=any


        Install Nvidia Driver

            curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_10.0.130-1_amd64.deb
            sudo dpkg -i cuda-repo-ubuntu1604_10.0.130-1_amd64.deb
            sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub

            sudo apt-get update
            sudo apt-get install cuda
        
            sudo pip3 install tensorflow-gpu


## Install requirements

run the installs from the dockerfile

    sudo apt-get update
    sudo apt-get install -y git libsm6 libxext6 libxrender1 python3-dev python3-pip

Note that specifying the python and pip versions is important.

    sudo pip3 install --upgrade pip
    sudo pip3 install Cython
    sudo pip3 install -r requirements_tpu.txt
    python3 setup.py build_ext --inplace


## Generating Data

    mkdir -p run1/train run1/val
 
    python3 make_tpu_dataset.py --phi 0 --split validation --path run1/val --debug --num-shards 20

   python3 make_tpu_dataset.py --phi 0 --split validation --path run1/train --debug --num-shards 20

    gsutil -m cp -r run1  gs://ondaka-ml-data/dev/run1
