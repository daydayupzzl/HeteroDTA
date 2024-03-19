# HeteroDTA: Deep Learning Model for Drug-Target Affinity Prediction

This repository contains the HeteroDTA deep learning model, which is designed for predicting drug-target affinity in drug discovery tasks. The model is implemented in Python and utilizes various deep learning techniques for accurate affinity predictions.

## Requirements
- numpy == 1.17.4 
- kreas == 2.3.1 
- pytorch == 1.8.0 
- matplotlib==3.2.2 
- pandas==1.2.4
- PyG (torch-geometric) == 1.3.2
- rdkit==2009.Q1-1
- tqdm==4.51.0
- numpy==1.20.1 
- scikit_learn==0.24.2 <br />

> Note: There are some dependencies that are not listed, please install them independently according to the feedback from the console

## Datasets
All publicly accessible datasets used can be accessed here:

| Dataset | link |
|----|----|
| Davis, KIBA| https://github.com/hkmztrk/DeepDTA/tree/master/data|
| Human and C.elegans | https://github.com/masashitsubaki/CPI_prediction|

## Usage
1. Clone the repository
    ``` shell
   git clone https://github.com/daydayupzzl/HeteroDTA.git
   cd your-repository
   ```
2. Install the required dependencies
    ``` shell
    pip install xxx # Please follow comsole's feedback and install the missing packages until there is no lack of dependencies
    ```
3. Training
    ``` shell
    python train.py --dataset your_dataset --options your_options
    ```
4. Inference
    ``` shell
    python train.py --dataset your_dataset --options your_options
    ```
> Please refer to the individual script files for more detailed instructions and options.

## Citation
If you use this code or related methods in your research, please consider citing HeteroDTA paper (We will add the literature address in the future)
``` text
    @article {,
        author = {},
        title = {},
        year={},
        publisher = {},
        journal = {}
    }
```