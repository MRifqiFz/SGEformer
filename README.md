# State-of-Health Prediction of Lithium-Ion Batteries using Exponential Smoothing Transformer with Seasonal and Growth Embedding

<p align="center">
<img src="./pics/SGEFormer Architecture.png" width = "700" alt="" align=center />
<br><br>
<b>Figure 1.</b> Overall SGEformer Architecture.
</p>

Official PyTorch code repository for the [SGEformer paper](https://doi.org/10.1109/ACCESS.2024.3357736).

## Requirements

1. Install Python 3.8, and the required dependencies.
2. Required dependencies can be installed by: ```pip install -r requirements.txt```

## Data

* Pre-processed datasets can be downloaded from the following
  links, [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/e1ccfff39ad541908bae/)
  or [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy?usp=sharing), as obtained
  from [Autoformer's](https://github.com/thuml/Autoformer) GitHub repository.
* Place the downloaded datasets into the `dataset/` folder, e.g. `dataset/ETT-small/ETTm2.csv`.

## Usage

1. Install the required dependencies.
2. Download data as above, and place them in the folder, `dataset/`.
3. Train the model. We provide the experiment scripts of all benchmarks under the folder `./scripts`,
   e.g. `./scripts/ETTm2.sh`. You might have to change permissions on the script files by running`chmod u+x scripts/*`.
4. The script for grid search is also provided, and can be run by `./grid_search.sh`.

## Citation
Please consider citing if you find this code useful to your research.
<pre>@article{fauzi2024state,
  title={State-of-Health Prediction of Lithium-Ion Batteries using Exponential Smoothing Transformer with Seasonal and Growth Embedding},
  author={Fauzi, Muhammad Rifqi and Yudistira, Novanto and Mahmudy, Wayan Firdaus},
  journal={IEEE Access},
  year={2024},
  publisher={IEEE}
}</pre>
