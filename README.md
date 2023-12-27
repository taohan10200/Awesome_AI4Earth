
# Awesome-ai4earth [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

![](resources/image8.gif)

🔥 Deep learning has been widely explored almost in all research filed. Here is a curated list of papers about deep learning  methods in Earth System, especially relating to weather prediction. It also contains frameworks for data-driven Numerical Weather Prediciton (NWP) training, tools to deploy weather prediction, courses and tutorials about Ai4earth and all publicly available weather prediction checkpoints and APIs.

## Table of Content
- [Awesome-ai4earth ](#Awesome-ai4earth)
  - [Updates](#updates)
  - [Milestone Papers](#milestone-papers)
  - [Preprocessing](#Preprocessing)
    - [Data Assimilation](#data-assimilation)
  - [Forcasting](#forecasting)
    - [Numerical Weather Prediciton](#numerical-weather-prediciton)
    - [Precipitation Nowcasting](#precipitation-prediction)
  - [Postprocessing](#postprocessing)
    - [Downscaling ](#downscaling)
    - [Bias Correction](#bias-correction)
  - [Applications](#application)
    - [Extreme Weather and Prediction](#extreme-weather-and-prediction)
      > tropical cyclones, heat wave etc.
    - [Climate phenomena analysis](#climate-phenomena-analysis)
  - [Weather&Climate Related Application](#weather-cliomate-related-application)
    - [global-wildfire-prediction](#global-wildfire-prediction)
  - [Other Papers](#other-papers)

  - [Ai4earth Benchmark](#ai4earth-leaderboard)
  - [Dataset](#dataset)
## Updates

- [2023-11-07] Creat this project, add some big model for atmospherical modeling!

### ToDos

- Add more paper, datasets, research directions for ai4earth :sparkles:**Contributions Wanted**

>  Also check out the project that I am currently working on: [EarthVision](https://github.com/taohan10200/EarthVision/tree/main/nwp) - A deep learniong framwork for Numerical Weather Prediction, Earth System Data cmpression, Precipitation Prediction)




## Milestone Papers for NWP

|  Date  |       keywords       |    Institute    | Paper                                                                                                                                                                               | Publication |
| :-----: | :------------------: | :--------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------: |
| 2022-02 |     FourCastNet     |      Nvidia      | FourCastNet: Accelerating Global High-Resolution Weather Forecasting Using Adaptive Fourier Neural Operators [[paper1](https://arxiv.org/abs/2202.11214)][[paper2](https://arxiv.org/abs/2202.11214)]|   PASC<br>  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F286aafc2453a3faeae539e939273938b690fa64a%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)<br>  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F10194e9d1d6b8ca8870445c990d4933c1dac1125%3Ffields%3DcitationCount&query=%24.citationCount&label=citation) |
| 2022-11 |     PanguWeather     |      HuaWei      | Accurate medium-range global weather forecasting with 3D neural networks [[paper1]](https://arxiv.org/pdf/1706.03762.pdf)[[paper2]](https://www.nature.com/articles/s41586-023-06185-3)|   Nature<br>  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F4405874879827cacf199d26e8e23e4f547f72a2c%3Ffields%3DcitationCount&query=%24.citationCount&label=citation) |
| 2022-12 |       GraphCast       |      DeepMind      | [GraphCast: Learning skillful medium-range global weather forecasting](https://arxiv.org/abs/2212.12794)| Science<br> ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F7dff3280beed4cef96265350074498bf142c41e7%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)          |
| 2023-04 |     FengWu     |      Shanghai AILab      | [FengWu: Pushing the Skillful Global Medium-range Weather Forecast beyond 10 Days Lead](https://arxiv.org/abs/2304.02948)|   arxiv<br>  ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Ff4a62db4dd86129561a16b0a18cc09985580554c%3Ffields%3DcitationCount&query=%24.citationCount&label=citation) |
| 2023-06 |     FuXi    |      Fudan University      | [FuXi: A cascade machine learning forecasting system for 15-day global weather forecast](e795f62df9ccac2a39e126f95404e5364d55193c) | npj <br>![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fe795f62df9ccac2a39e126f95404e5364d55193c%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)    |
# Preprocessing
## data-assimilation
- [FengWu-Adas](https://arxiv.org/abs/2312.12462) Towards an End-to-End Artificial Intelligence Driven Global Weather Forecasting System. (arxiv, 12/2023) 
- [FengWu-4DVar](https://arxiv.org/abs/2312.12455) FengWu-4DVar: Coupling the Data-driven Weather Forecasting Model with 4D Variational Assimilation  (arxiv, 12/2023)

# Forcasting
## numerical-weather-prediction 
- [NeuralGCM](https://arxiv.org/abs/2311.07222) Neural General Circulation Models. (arxiv, 11/2023) 
- [GenCast](https://arxiv.org/pdf/2312.15796.pdf) Diffusion-based ensemble forecasting
for medium-range weather. (arxiv. 12/2023)
- [MetNet-3](https://arxiv.org/abs/2306.06079) Deep Learning for Day Forecasts from Sparse Observations. (arxiv, 06/2023) 
- [ClimaX](https://arxiv.org/abs/2301.10343) A foundation model for weather and climate, (ICML, 2023) [[project]](https://github.com/microsoft/ClimaX)

## precipitation-prediction
- [NowcastNet](https://www.nature.com/articles/s41586-023-06184-4) Skilful nowcasting of extreme precipitation with NowcastNet. (nature, 07/2023)
- [PostRainBench](https://arxiv.org/abs/2310.02676) - A comprehensive benchmark and a new model for precipitation forecasting. (arxiv, 10/2023)
- [Anthropogenic fingerprints in daily precipitation revealed by deep learning](https://www.nature.com/articles/s41586-023-06474-x). (nature, 08/2023)


# Postprocessing
  ## downscaling
  - [CorrDiff](https://arxiv.org/abs/2309.15214) Generative residual diffusion modeling for km-scale atmospheric downscaling, (arxiv, 09/2023)
  ## bias-correction
  - Two deep learning-based bias-correction pathways improve summer precipitation prediction over China, (Environmental Research Letters, 12/2022) [paper](https://iopscience.iop.org/article/10.1088/1748-9326/aca68a/pdf) 

 # Applications
  ## extreme-weather-and-prediction
  - [FuXi-Extreme](https://arxiv.org/abs/2310.19822): Improving extreme rainfall and wind forecasts with diffusion mode. (arxiv, 10/2023)

# weather-cliomate-related-application
## global-wildfire-prediction
- [The Potential Predictability of Fire Danger Provided by Numerical Weather Prediction](https://journals.ametsoc.org/configurable/content/journals$002fapme$002f55$002f11$002fjamc-d-15-0297.1.xml?t%3Aac=journals%24002fapme%24002f55%24002f11%24002fjamc-d-15-0297.1.xml&tab_body=pdf) (JAMC, 11/2016)
- [Machine learning–based observation-constrained projections reveal elevated global socioeconomic risks from wildfire.](https://www.nature.com/articles/s41467-022-28853-0) (nature communication, 22/03/2023)

#  ai4earth-benchmark
[WeatherBench 2](https://arxiv.org/abs/2308.15560): A benchmark for the next generation of data-driven global weather models [[project]](https://github.com/google-research/weatherbench2) [[LeaderBoard]](https://sites.research.google/weatherbench/) (08/2023)
## Other Papers
If you're interested in the field of AI4Earth, you may find the above list of milestone papers helpful to explore its history and state-of-the-art. However, each direction of AI4Earth offers a unique set of insights and contributions, which are essential to understanding the field as a whole. For a detailed list of papers in various subfields, please refer to the following link (it is possible that there are overlaps between different subfields):

(:exclamation: **We would greatly appreciate and welcome your contribution to the following list. :exclamation:**)
- [ML-weather-Survey](paper_list/weather_survey.md)

- [Ai4earth-NWP-Analysis](paper_list/nwp_analysis.md)

  > Analyse different NWP models in different fields with respect to different abilities

- [Ai4earth-Wildfire](paper_list/Wildfire-Related)

  > some deeplearning methods in wildfire forecast.

- [Ai4earth-Application](paper_list/application.md)


## Weather Forecast Leaderboard
<!-- <div align=center>
<img src="resources/creepy_llm.jpeg" width="500">
</div> -->

The following list makes sure that all weather forecasting models are compared **apples to apples**.
  > You may also find these leaderboards helpful:
  > - [WeatherBench2 Leaderboard](https://sites.research.google/weatherbench/) - evaluating and comparing various weather forecasting models. displays up-to-date scores of many of the state-of-the-art ML and physics-based models.


### Base Weather Forecasting Models

|       Model       | Size |  Architecture  |                                                                                               Access                                                                                               |  Date  | Origin                                                                                                                                       | Model License[^1] |
| :----------------: | :--: | :-------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----: | -------------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| PanguWeather | ~740M |  3D Transformer  | [ckpt](https://github.com/198808xc/Pangu-Weather)| 2022-11 | [Paper](https://arxiv.org/pdf/1706.03762.pdf)| [BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) |
|    GraphCast      | unkonwn |  GNN  |  [ckpt](https://github.com/google-deepmind/graphcast)  | 2022-12 | [Paper](https://arxiv.org/abs/2212.12794)| [Apache 2.0](https://github.com/yandex/YaLM-100B/blob/14fa94df2ebbbd1864b81f13978f2bf4af270fcb/LICENSE) & [BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) |
|    FengWu      | unkonwn |  U-Transformer  |  [ckpt](https://drive.google.com/drive/folders/1NhrcpkWS6MHzEs3i_lsIaZsADjBrICYV)  | 2023-06 | [Paper](https://arxiv.org/pdf/2306.12873.pdf)| - |
|    FuXi      | unkonwn |  Transformer  |  [ckpt](https://github.com/OpenEarthLab/FengWu)  | 2023-04 | [Paper](https://arxiv.org/abs/2304.02948)| - |



## BigModel Training Frameworks

- [DeepSpeed](https://github.com/microsoft/DeepSpeed) - DeepSpeed is a deep learning optimization library that makes distributed training and inference easy, efficient, and effective.
- [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed) - DeepSpeed version of NVIDIA's Megatron-LM that adds additional support for several features such as MoE model training, Curriculum Learning, 3D Parallelism, and others. 
- [FairScale](https://fairscale.readthedocs.io/en/latest/what_is_fairscale.html) - FairScale is a PyTorch extension library for high performance and large scale training.
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) - Ongoing research training transformer models at scale.
- [Colossal-AI](https://github.com/hpcaitech/ColossalAI) - Making large AI models cheaper, faster, and more accessible.
- [BMTrain](https://github.com/OpenBMB/BMTrain) - Efficient Training for Big Models.
- [Mesh Tensorflow](https://github.com/tensorflow/mesh) - Mesh TensorFlow: Model Parallelism Made Easier.
- [maxtext](https://github.com/google/maxtext) - A simple, performant and scalable Jax LLM!
- [Alpa](https://alpa.ai/index.html) - Alpa is a system for training and serving large-scale neural networks.
- [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) - An implementation of model parallel autoregressive transformers on GPUs, based on the DeepSpeed library.

## Tools for deploying Weather Forecasting Models


- [Fluid-Earth](https://fluid-earth.byrd.osu.edu/#date=2023-12-27T10%3A00%3A00.000Z&gdata=temperature+at+2+m+above+ground&pdata=wind+at+10+m+above+ground&proj=vertical+perspective&lat=26.81&lon=80.85&zoom=1.50&smode=true&kmode=false&pins=%5B%5D) - An interactive web application that allows you to visualize current and past conditions of Earth's atmosphere and oceans.




## Other Awesome Lists

- [TODO](xx) - A curated (still actively updated) list of practical guide resources of earth-related research.

## Contributing

This is an active repository and your contributions are always welcome!

I will keep some pull requests open if I'm not sure if they are awesome for ai4earth, you could vote for them by adding 👍 to them.

---

If you have any question about this opinionated list, do not hesitate to contact me hantao10200@gmail.com.

[^1]: This is not legal advice. Please contact the original authors of the models for more information.