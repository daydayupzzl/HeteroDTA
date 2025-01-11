# <center> HeteroDTA: Deep Learning Model for Drug-Target Affinity Prediction </center>

[![Python 3.7](https://img.shields.io/badge/Python-3.7-blue)]()
[![PyTorch 1.10+](https://img.shields.io/badge/PyTorch-1.10%2B-red)]()
[![ESM](https://img.shields.io/badge/ESM%2B-green)]()
[![rdkit 2023.3.2+](https://img.shields.io/badge/rdkit-2023.3.2%2B-purple)]()
[![torch-geometric 2.3.1+](https://img.shields.io/badge/torch--geometric-2.3.1%2B-yellow)]()

<p align="center">
  <img src="png\avatar.png" width="30%" alt="Avatar" style="border-radius: 20px; ">
</p>

## 🥰 Introduction
This repository contains the HeteroDTA deep learning model, which is designed for predicting drug-target affinity in drug discovery tasks. The model is implemented in Python and utilizes various deep learning techniques for accurate affinity predictions.


## :satisfied: Requirements
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

> :construction: Note: There are some dependencies that are not listed, please install them independently according to the feedback from the console

## :rainbow: Datasets

All publicly accessible datasets used can be accessed here:

| Dataset Name        | Link                                                |
|---------------------|-----------------------------------------------------|
| Davis, KIBA         | https://github.com/hkmztrk/DeepDTA/tree/master/data |
| Human and C.elegans | https://github.com/masashitsubaki/CPI_prediction    |

> :construction: Note: It is necessary to download the corresponding data set and then place it in the corresponding directory named "data" on your own machine for subsequent data preprocessing

## :rocket: Pre-trained model

All publicly accessible models used can be accessed here:

| Model Name | Link                                        |
|------------|---------------------------------------------|
| GEM        | https://github.com/PaddlePaddle/PaddleHelix |
| ESM        | https://github.com/facebookresearch/esm     |

> :construction: Note: It is necessary to read the corresponding GitHub guidelines and use pre-trained models on your own machine to preprocess your chosen datasets or your own private datasets

- The role of GEM: obtain atomic embeddings in compounds
- The role of ESM: (a) predict protein structure; (b) obtain embeddings of amino acids in proteins

## :aerial_tramway: Install tutorial

> :construction: Notes: 
> - Before training or inference, the datasets used must be preprocessed 
> - Please refer to the individual script files for more detailed instructions and options.

1. Clone the repository
    ``` shell
   git clone https://github.com/daydayupzzl/HeteroDTA.git
   cd your-repository
   ```
2. Install the required dependencies
    ``` shell
    pip install SomePackage # Please follow comsole's feedback and install the missing packages until there is no lack of dependencies
    ```
## :page_with_curl: Training tutorial

1. Make sure you have the necessary Python libraries installed, including PyTorch.
2. Open a terminal (Command Prompt on Windows, or Terminal on Mac/Linux).
3. Navigate to the directory containing the file.
4. Run the following command:

    ``` shell
    python training.py <dataset_index> <cuda_index> <dataset_type_index>
    ```
   Replace <dataset_index>, <cuda_index>, and <dataset_type_index> with the following values:
   
   - dataset_index: Dataset index. 0 for 'davis', 1 for 'kiba'.
   - cuda_index: GPU index. 0 for 'cuda:0', 1 for 'cuda:1'.
   - dataset_type_index: Dataset type index. 0 for 'original', 1 for 'cold_drug', 2 for 'cold_protein', 3 for 'cold_pair'.
   
   For example:
   ``` shell
   python your_script_name.py 0 0 0 v1
    ```
   This will run the script, using the "davis" dataset, utilizing the first GPU, and employing the original dataset type.

## :stuck_out_tongue_closed_eyes: Inference tutorial

1. Ensure that you have trained one model
2. Open a terminal (Command Prompt on Windows, or Terminal on Mac/Linux). 
3. Navigate to the directory containing the file.
4. Run the following command:
   ``` shell
   python inference.py <dataset_index> <cuda_index> <dataset_type_index>
    ```
   Replace <dataset_index>, <cuda_index>, and <dataset_type_index> with the following values:
   - dataset_index: Dataset index. 0 for 'davis', 1 for 'kiba'.
   - cuda_index: GPU index. 0 for 'cuda:0', 1 for 'cuda:1', 2 for 'cuda:2', 3 for 'cuda:3'.
   - dataset_type_index: Dataset type index. 0 for 'original', 1 for 'cold_drug', 2 for 'cold_protein', 3 for 'cold_pair'.
   
   For example:
   ``` shell
   python inference.py 0 0 0
    ```
   This will execute the script, using the "davis" dataset, utilizing the first GPU, and employing the original dataset type.
   
## :heartpulse: Citation

If you use this code or related methods in your research, please consider citing HeteroDTA paper (We will add the literature address in the future)
``` text
@article{10.1093/bioinformatics/btae240,
    author = {Zhang, Zuolong and He, Xin and Long, Dazhi and Luo, Gang and Chen, Shengbo},
    title = {Enhancing generalizability and performance in drug–target interaction identification by integrating pharmacophore and pre-trained models},
    journal = {Bioinformatics},
    volume = {40},
    number = {Supplement_1},
    pages = {i539-i547},
    year = {2024},
    month = {06},
    issn = {1367-4811},
    doi = {10.1093/bioinformatics/btae240},
    url = {https://doi.org/10.1093/bioinformatics/btae240},
    eprint = {https://academic.oup.com/bioinformatics/article-pdf/40/Supplement\_1/i539/58355122/btae240.pdf},
}
```
