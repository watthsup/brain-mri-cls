# Brain Tumor Classification using MRI Images

This project is a deep learning-based approach to classify four types of brain tumors from MRI images: meningioma, glioma, pituitary, and no tumor. The goal of this project is to provide an accurate and automated method for brain tumor diagnosis.

## Dataset
The dataset used in this project can be found on Kaggle: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

## Features

- Classifies four types of brain tumors from MRI images with high accuracy.
- Utilizes PyTorch to train and evaluate several CNN backbones.
- Offers a notebook for exploratory data analysis (EDA) and model inference, as well as a src folder for training the model.
- Includes a config.ini file to customize the hyperparameters for the model.
- The source code is well-organized, making it easy to adapt to other deep learning classification problems. There are only few parts you need to change i.e. dataset.py

## Installation

To install and run the application, please follow the steps below:

1. Clone the repository to your local machine.
2. Install the required Python packages by running `pip install -r requirements.txt`.
3. Run the `EDA.ipynb` notebook in the notebook folder to visualize the dataset and explore the data.
4. Modify the `config.ini` file to customize the hyperparameters for the model.
5. Train the model by running `python train.py` in the `src` folder.
6. Evaluate the model's performance by running `inference.ipynb` notebook in the `notebook` folder.

## Usage

After following the installation steps above, you can use the application to classify brain tumors from MRI images. You can customize the hyperparameters in the `config.ini` file to achieve better performance.

## Contributing

Contributions to Brain Tumor Classification are welcome! Please refer to the `CONTRIBUTING.md` file for guidelines on how to contribute.

## License

This project is licensed under the MIT License. Please refer to the `LICENSE` file for more information.
