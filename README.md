## Introduction

#### This repository provides the implemetation of the replayed speech detection system used in the papers:

Shih-Kuang Lee, Yu Tsao, and Hsin-Min Wang, “[A Study of Using Cepstrogram for Countermeasure Against Replay Attacks](https://arxiv.org/abs/2204.04333),” arXiv preprint arXiv:2204.04333, 2022.

Shih-Kuang Lee, Yu Tsao, and Hsin-Min Wang, “[Detecting Replay Attacks Using Single-Channel Audio: The Temporal Autocorrelation of Speech](https://homepage.iis.sinica.edu.tw/papers/whm/25385-F.pdf),” in 2022 Asia Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC) (APSIPA ASC 2022), Chiang Mai, Thailand, Nov. 2022.

## Results

All systems reported in the papers are trained, validated and evaluated on [ASVspoof 2019 physical access (PA)](https://www.asvspoof.org/index2019.html) database, scores can be found in the directory [CMs](./CMs).

#### Results of the baseline system, it is the implementation of the system used in:

Lavrentyeva, G., Novoselov, S., Tseren, A., Volkova, M., Gorlanov, A., Kozlov, A. (2019) [STC Antispoofing Systems for the ASVspoof2019 Challenge](https://www.isca-speech.org/archive/interspeech_2019/lavrentyeva19_interspeech.html). Proc. Interspeech 2019, 1033-1037, doi: 10.21437/Interspeech.2019-1768

![](./CMs/Results-Baseline.png "Baseline system")

#### Results of the single systems:

![](./CMs/Results-Single.png "Single systems")

#### Results of the fusion systems fused with equal weight (sum of scores):

![](./CMs/Results-Fusion.png "Fusion systems")

## Dependencies
```
pip install -r requirements.txt
```

## Prepare data
```
sh prepare_PA.sh
```

## Training, validation and evaluation
```
python main.py --cuda_vd 0 --path_data ../ASVspoof2019 --task PA --conifg_section Ceps
python main.py --cuda_vd 0 --path_data ../ASVspoof2019 --task PA --conifg_section TAC --dmode_train fixed --dmode___dev fixed --dmode__eval fixed
```

## Citation Information

Shih-Kuang Lee, Yu Tsao, and Hsin-Min Wang, “A Study of Using Cepstrogram for Countermeasure Against Replay Attacks,” arXiv preprint arXiv:2204.04333, 2022.
```bibtex
@article{lee2022study,
  title={{A Study of Using Cepstrogram for Countermeasure Against Replay Attacks}},
  author={Shih-Kuang Lee and Yu Tsao and Hsin-Min Wang},
  journal={arXiv preprint arXiv:2204.04333},
  year={2022}}
```

Shih-Kuang Lee, Yu Tsao, and Hsin-Min Wang, “Detecting Replay Attacks Using Single-Channel Audio: The Temporal Autocorrelation of Speech,” in 2022 Asia Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC) (APSIPA ASC 2022), Chiang Mai, Thailand, Nov. 2022.
```bibtex
@INPROCEEDINGS{Lee2211:Detecting,
  AUTHOR={Shih-Kuang Lee and Yu Tsao and Hsin-Min Wang},
  TITLE={{Detecting Replay Attacks Using Single-Channel Audio: The Temporal Autocorrelation of Speech}},
  BOOKTITLE={2022 Asia Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC) (APSIPA ASC 2022)},
  ADDRESS={Chiang Mai, Thailand},
  MONTH={nov},
  YEAR={2022}}
```

## Licensing

This repository is licensed under the [ISC License](https://github.com/shihkuanglee/RD-LCNN/blob/main/LICENSE.md).

This repository includes modified codes from [Librosa](https://github.com/librosa/librosa), also ISC licensed.
