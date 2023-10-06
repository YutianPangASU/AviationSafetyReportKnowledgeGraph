# NSTB Aviation Safety Report Knowledge Base

## Environment
Assume you have conda installed,
```
conda create -n ntsb python=3.9
conda activate ntsb
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c conda-forge transformers
pip install wikipedia newspaper3k GoogleNews pyvis
```

## Data Accessibility
To download the data from NSTB website,
```
wget https://data.ntsb.gov/avdata/FileDirectory/DownloadFile?fileID=C%3A%5Cavdata%5Cavall.zip
unzip DownloadFile\?fileID\=C\:\\avdata\\avall.zip 
rm -rf DownloadFile\?fileID\=C\:\\avdata\\avall.zip
ls -lh
```

The ```avall.mdb``` Microsoft Access database format is tricky to run on Linux OS. 

```
sudo apt install mdbtools
pip install pandas_access
```


## Build the NTSB Safety Knowledge Graph
e