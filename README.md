
# GPT4EQ : LLM for Earthquake Monitoring

## Introduction
GPT4EQ is a deep learning model using LLM for seismic signal processing, which can be used for seismic monitoring tasks including event detection and phase picking for P/S waves.

The few-shot learning task is conducted owing to constraints on computational resources. Alternatively, the zero-shot task can be tested on a different dataset, although the results tend to be less satisfactory.

<div align="center"><img src=./images/structure.png width=60% /></div>

## Usage
### Install
```
git clone https://github.com/guanlii/GPT4EQ
cd GPT4EQ
pip install -r requirements.txt
```

### Data preparation

- **Data download**
  
  Create a new folder in the directory`datasets` named `STEAD` and download `chunk2` data from [STEAD Dataset](https://github.com/smousavi05/STEAD). The file structure is as following.

  ```Shell
  ├── datasets
  │   ├── STEAD
  │   │   ├── chunk2.hdf5
  │   │   ├── chunk2.csv
  ```  

- **For model deployment**

  Run the notebook `demo_predictgpt.ipynb` and replace the trace_name.

### Training


- **Few shot training & testing**<br/>
  Use the following command to start few shot training & testing:

  ```Shell
  python main.py \
    --seed 0 \
    --mode "train_test" \
    --model-name "GPT4EQ" \
    --log-base "./logs" \
    --device "cuda:0" \
    --data "./datasets/STEAD/" \
    --dataset-name "stead" \
    --data-split True \
    --train-size 0.8 \
    --val-size 0.1 \
    --shuffle True \
    --in-samples 6000 \
    --workers 4 \
    --augmentation True \
    --epochs 200 \
    --patience 30 \
    --batch-size 20 \
    --few-shot-ratio 0.05 \
    --pretrain True \
    --gpt-layers 6
  ```

### Testing
  The pretrained weight can be download [here](https://drive.google.com/file/d/1VkVX03KqlItvjkHN1jgH3xz_gp_K-hap/view?usp=sharing). The file structure is as following.
  ```
  ├── logs
  │   ├── example
  │   │   ├── checkpoints
  │   │   │   ├── GPT4EQ.pth
  ```
  
  Use the following command to start testing only:

  ```Shell
  python main.py \
    --seed 0 \
    --mode "test" \
    --model-name "GPT4EQ" \
    --log-base "./logs" \
    --checkpoint "./logs/example/checkpoints/GPT4EQ.pth" \
    --device "cuda:0" \
    --data "./datasets/STEAD/" \
    --dataset-name "stead" \
    --data-split True \
    --train-size 0.8 \
    --val-size 0.1 \
    --in-samples 6000 \
    --workers 4 \
    --batch-size 20 \
    --few-shot-ratio 0.05
  ```

  For zero shot task testing, here is an example. You can get PNW data from [here](https://github.com/niyiyu/PNW-ML) and place it as like how STEAD dataset is place, then use the following command to start testing:

  ```Shell
  python main.py \
    --seed 0 \
    --mode "test" \
    --model-name "GPT4EQ" \
    --log-base "./logs" \
    --checkpoint "./logs/example/checkpoints/GPT4EQ.pth" \
    --device "cuda:0" \
    --data "./datasets/PNW/" \
    --dataset-name "pnw" \
    --data-split True \
    --train-size 0 \
    --val-size 0 \
    --in-samples 6000 \
    --workers 4 \
    --batch-size 20
  ```
  You can also test it by using Google Colab.[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_Zo5YFQ8BROrKp4j2-Masb4vDFqEbV6B?usp=sharing)



## Acknowledgement
This benchmark has been adapted from [SeisT](https://github.com/senli1073/SeisT), an excellent work on deep learning models for earthquakes monitoring.
```
@ARTICLE{10453976,
  author={Li, Sen and Yang, Xu and Cao, Anye and Wang, Changbin and Liu, Yaoqi and Liu, Yapeng and Niu, Qiang},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={SeisT: A Foundational Deep-Learning Model for Earthquake Monitoring Tasks}, 
  year={2024},
  volume={62},
  pages={1-15},
  doi={10.1109/TGRS.2024.3371503}
}
```

The framework for processing LLM has been adapted from [GPT4TS](https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All), an pioneering work in LLM for time series.
```
@inproceedings{zhou2023onefitsall,
  title={{One Fits All}: Power General Time Series Analysis by Pretrained LM},
  author={Tian Zhou, Peisong Niu, Xue Wang, Liang Sun, Rong Jin},
  booktitle={NeurIPS},
  year={2023}
}
```


This work is based on the following excellent works.

- SeisT: A Foundational Deep-Learning Model for Earthquake Monitoring Tasks [[paper](https://doi.org/10.1109/TGRS.2024.3371503)] [[code](https://github.com/senli1073/SeisT)]

- One Fits All: Power General Time Series Analysis by Pretrained LM [[paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/86c17de05579cde52025f9984e6e2ebb-Abstract-Conference.html)] [[code](https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All)]

- Earthquake transformer—an attentive deep-learning model for simultaneous earthquake detection and phase picking [[paper](https://www.nature.com/articles/s41467-020-17591-w)] [[code](https://github.com/smousavi05/EQTransformer)]

- PhaseNet: A deep-neural-network-based seismic arrival-time picking method [[paper](https://academic.oup.com/gji/article/216/1/261/5129142)] [[code](https://github.com/AI4EPS/PhaseNet)]

## License
Licensed under an MIT license.



