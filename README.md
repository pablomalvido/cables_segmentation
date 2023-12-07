# Generation of realistic synthetic cable images to train AI segmentation models

This package must contain the following files/folders:

- blender_script/: This folder contains the python script used in Blender to generate the images for the synthetic cables segmentation dataset.

- environments/: This folder contains two anaconda environments:
	- blender.yml: This is the environment that has to be linked with Blender's Python to run the blender_script/synthetic_cables_generation.py script.
	- tf3env.yml: This is the environment that must be used to test the trained segmentation models with the segment_all.py and the segment_one.py scripts.
	
- models/: This folder contains the trained segmentation models. The models can be downloaded from https://drive.google.com/drive/folders/1jXdS53sfyJ7MOfui93D3Cl22wtmTkyXU?usp=sharing. After downloading them, add the models/ folder to the package.

- real_images/: This folder contains the 54 real cable images and the ground truth segmentation masks (labelled manually) that have been used to validate the models trained with the synthetic dataset. Additionally, this molder contains the resulting segmentation with each of the trained models.

- synthetic_images/: This folder contains some of the synthetic images and masks utilized to train the models (180 images, 30 for each category). After the paper is published, a link will be added to download all the images from the dataset (25.000 images).

- segment_all.py: Python script that segments all the images in a folder with all the trained models and saves the resulting images. The user must specify the relative path of the directory (e.g., "real_images"). An example of this is already included in the script.

- segment_one.py: Python script that segments a single image with one of the trained models and displays the resulting image. The user must specify the relative path of the directory (e.g., "real_images"), the image_name (e.g., "20231108_143246_7.jpg"), and the model ("unet"). An example of this is already included in the script.

## Usage
1. Use the link provided in the models/ folder description to download the models trained with the synthetic dataset. Add the downloaded models/ folder to the package.

2. Import and activate the proper anaconda environment depending on the script you want to use, as described in the environments/ folder description. To create and activate the environment follow these steps: (i) Install Anaconda, (ii) Create environment from yaml file, e.g., ``conda env create -f tf3env.yml``, (iii) Activate the environment, e.g., ``conda activate tf3env``.

3. Run the segment_all.py or segment_one.py script to segment images with the trained models.

## Additional considerations
If you want to use your own images to segment cables, the resolution of the images must be 512x512.
