Will be updated on 4.15.2022.

# RD-LCNN
This repository provides the replay detection systems used in ['A Study of Using Cepstrogram for Countermeasure Against Replay Attacks'](https://arxiv.org/abs/2204.04333)

### Dependencies
```
pip install -r requirements.txt
```

### Training & Testing
```
python main.py --cuda_vd 0 --epochs 200 --esep 100 --conifg_section Ceps --lr -1 --path_data ../ASVspoof2019 --task PA
```

