---
comments: true
description: Follow this comprehensive guide to set up and operate YOLOv5 on an AWS Deep Learning instance for object detection tasks. Get started with model training and deployment.
keywords: YOLOv5, AWS Deep Learning AMIs, object detection, machine learning, AI, model training, instance setup, Ultralytics
---

# YOLOv5 🚀 on AWS Deep Learning Instance: Your Complete Guide

Setting up a high-performance deep learning environment can be daunting for newcomers, but fear not! 🛠️ With this guide, we'll walk you through the process of getting YOLOv5 up and running on an AWS Deep Learning instance. By leveraging the power of Amazon Web Services (AWS), even those new to machine learning can get started quickly and cost-effectively. The AWS platform's scalability is perfect for both experimentation and production deployment.

Other quickstart options for YOLOv5 include our [Colab Notebook](https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb) <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> <a href="https://www.kaggle.com/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>, [GCP Deep Learning VM](https://docs.ultralytics.com/yolov5/environments/google_cloud_quickstart_tutorial), and our Docker image at [Docker Hub](https://hub.docker.com/r/ultralytics/yolov5) <a href="https://hub.docker.com/r/ultralytics/yolov5"><img src="https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker" alt="Docker Pulls"></a>.

## Step 1: AWS Console Sign-In

Start by creating an account or signing in to the AWS console at [https://aws.amazon.com/console/](https://aws.amazon.com/console/). Once logged in, select the **EC2** service to manage and set up your instances.

![Console](https://user-images.githubusercontent.com/26833433/106323804-debddd00-622c-11eb-997f-b8217dc0e975.png)

## Step 2: Launch Your Instance

In the EC2 dashboard, you'll find the **Launch Instance** button which is your gateway to creating a new virtual server.

![Launch](https://user-images.githubusercontent.com/26833433/106323950-204e8800-622d-11eb-915d-5c90406973ea.png)

### Selecting the Right Amazon Machine Image (AMI)

Here's where you choose the operating system and software stack for your instance. Type 'Deep Learning' into the search field and select the latest Ubuntu-based Deep Learning AMI, unless your needs dictate otherwise. Amazon's Deep Learning AMIs come pre-installed with popular frameworks and GPU drivers to streamline your setup process.

![Choose AMI](https://user-images.githubusercontent.com/26833433/106326107-c9e34880-6230-11eb-97c9-3b5fc2f4e2ff.png)

### Picking an Instance Type

For deep learning tasks, selecting a GPU instance type is generally recommended as it can vastly accelerate model training. For instance size considerations, remember that the model's memory requirements should never exceed what your instance can provide.

**Note:** The size of your model should be a factor in selecting an instance. If your model exceeds an instance's available RAM, select a different instance type with enough memory for your application.

For a list of available GPU instance types, visit [EC2 Instance Types](https://aws.amazon.com/ec2/instance-types/), specifically under Accelerated Computing.

![Choose Type](https://user-images.githubusercontent.com/26833433/106324624-52141e80-622e-11eb-9662-1a376d9c887d.png)

For more information on GPU monitoring and optimization, see [GPU Monitoring and Optimization](https://docs.aws.amazon.com/dlami/latest/devguide/tutorial-gpu.html). For pricing, see [On-Demand Pricing](https://aws.amazon.com/ec2/pricing/on-demand/) and [Spot Pricing](https://aws.amazon.com/ec2/spot/pricing/).

### Configuring Your Instance

Amazon EC2 Spot Instances offer a cost-effective way to run applications as they allow you to bid for unused capacity at a fraction of the standard cost. For a persistent experience that retains data even when the Spot Instance goes down, opt for a persistent request.

![Spot Request](https://user-images.githubusercontent.com/26833433/106324835-ac14e400-622e-11eb-8853-df5ec9b16dfc.png)

Remember to adjust the rest of your instance settings and security configurations as needed in Steps 4-7 before launching.

## Step 3: Connect to Your Instance

Once your instance is running, select its checkbox and click Connect to access the SSH information. Use the displayed SSH command in your preferred terminal to establish a connection to your instance.

![Connect](https://user-images.githubusercontent.com/26833433/106325530-cf8c5e80-622f-11eb-9f64-5b313a9d57a1.png)

## Step 4: Running YOLOv5

Logged into your instance, you're now ready to clone the YOLOv5 repository and install dependencies within a Python 3.8 or later environment. YOLOv5's models and datasets will automatically download from the latest [release](https://github.com/ultralytics/yolov5/releases).

```bash
git clone https://github.com/ultralytics/yolov5  # clone repository
cd yolov5
pip install -r requirements.txt  # install dependencies
```

With your environment set up, you can begin training, validating, performing inference, and exporting your YOLOv5 models:

```bash
# Train a model on your data
python train.py

# Validate the trained model for Precision, Recall, and mAP
python val.py --weights yolov5s.pt

# Run inference using the trained model on your images or videos
python segment.py --weights yolov5s.pt --source path/to/images

# Export the trained model to other formats for deployment
python export.py --weights yolov5s.pt --include onnx coreml tflite
```

## Optional Extras

To add more swap memory, which can be a savior for large datasets, run:

```bash
sudo fallocate -l 64G /swapfile  # allocate 64GB swap file
sudo chmod 600 /swapfile  # modify permissions
sudo mkswap /swapfile  # set up a Linux swap area
sudo swapon /swapfile  # activate swap file
free -h  # verify swap memory
```

And that's it! 🎉 You've successfully created an AWS Deep Learning instance and run YOLOv5. Whether you're just starting with object detection or scaling up for production, this setup can help you achieve your machine learning goals. Happy training, validating, and deploying! If you encounter any hiccups along the way, the robust AWS documentation and the active Ultralytics community are here to support you.
