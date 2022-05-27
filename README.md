This repository provides the replay detection systems used in
['A Study of Using Cepstrogram for Countermeasure Against Replay Attacks'](https://arxiv.org/abs/2204.04333)

### Dependencies
```
pip install -r requirements.txt
```

### Training & Testing
```
python main.py --cuda_vd 0 --path_data ../../ASVspoof2019 --task PA --conifg_section Ceps
```

### Licensing

This repository is licensed under the [ISC License](https://github.com/shihkuanglee/RD-LCNN/blob/main/LICENSE.md).

This repository includes modified code from [Librosa](https://github.com/librosa/librosa), also ISC licensed.
