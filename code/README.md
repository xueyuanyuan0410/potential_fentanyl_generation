Title: Exploring the fentanyl chemical space by using deep learning models

Owing to the sensitivity of the data and the potential for misuse, original fentanyl and its analogue data are not available to the public for unrestricted download. A demonstration dataset of 100 SMILES strings for drug-like small molecules sampled at random from the database of Moses is provided at potential_fentanyl_generation\potential_fentanyl_generation\code\1、encoding\data.csv. You can use this data to try and experience how the code works.

# 1、encoding
The code in this file converts the smiles format of molecules into SeqGAN training data.

# 2、seqGAN
The environment configured for this code is in the tf1_py2.yml.
In this file is the code for SeqGAN.
 # Training & Generation
 conda activate tf1_py2
 cd fentanyl analogues generation/code/2、seqGAN/SeqGAN-1
 python sequence_gan.py

# 3、decoding
The code in this file converts SeqGAN generation data into the smiles format of molecules.

# 4、screening valid molecules
The code in this file can pick out the valid molecules from the generated molecules.

# 5、draw molecular picture
The code in this file converts the smiles format for generating molecules into molecular diagrams.

# 6、Molecular similarity
The code in this file can calculate molecular similarity and draw similarity heatmaps.

# 7、Molecular properties and distribution
The code in this file can calculate the properties of molecules and draw the distibution of properties of molecules.

# 8、data augmentation
The code in this file can do data augmentation.

# 9、remove duplicate molecules
The code in this file can remove duplicate molecules.

# 10、MolGPT
In this work, we train small custom GPT on Moses and Guacamol dataset with next token prediction task. The model is then used for conditional molecular generation. 

- The processed Guacamol and MOSES datasets in csv format can be downloaded from this link:

https://drive.google.com/drive/folders/1LrtGru7Srj_62WMR4Zcfs7xJ3GZr9N4E?usp=sharing

- Original Guacamol dataset can be found here:

https://github.com/BenevolentAI/guacamol

- Original Moses dataset can be found here:

https://github.com/molecularsets/moses

- All trained weights can be found here:

https://www.kaggle.com/virajbagal/ligflow-final-weights


To train the model, make sure you have the datasets' csv file in the same directory as the code files.

# Training

```
./train_moses.sh
```

```
./train_guacamol.sh
```

# Generation

```
./generate_guacamol_prop.sh
```

```
./generate_moses_prop_scaf.sh
```

If you find this work useful, please cite:

Bagal, Viraj; Aggarwal, Rishal; Vinod, P. K.; Priyakumar, U. Deva (2021): MolGPT: Molecular Generation using a Transformer-Decoder Model. ChemRxiv. Preprint. https://doi.org/10.26434/chemrxiv.14561901.v1 


