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
  links, [Google Drive](https://drive.google.com/drive/folders/19-F1TusCgXMsqzgDji1qIwh5PgAmTjNp?usp=drive_link), as obtained
  from NASA Battery Dataset and CALCE Battery Dataset.
* Place the downloaded datasets into the `dataset/` folder, e.g. `dataset/battery/B0005_rev.csv`.

## Usage

1. Install the required dependencies.
2. Download data as above, and place them in the folder, `dataset/`.
3. Train the model. We provide the experiment scripts of all benchmarks under the folder `./scripts`,
   e.g. `./scripts/battery.sh`. You might have to change permissions on the script files by running`chmod u+x scripts/*`.

## Citation
Please consider citing if you find this code useful to your research.
<pre>@article{fauzi2024state,
  title={State-of-Health Prediction of Lithium-Ion Batteries using Exponential Smoothing Transformer with Seasonal and Growth Embedding},
  author={Fauzi, Muhammad Rifqi and Yudistira, Novanto and Mahmudy, Wayan Firdaus},
  journal={IEEE Access},
  year={2024},
  publisher={IEEE}
}</pre>
