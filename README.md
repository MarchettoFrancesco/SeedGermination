# Seed Germination — Computer Vision 

Repository per classificazione automatica della germinazione dei semi (classe 0 = non germinato, classe 1 = germinato) 
partendo da immagini acquisite con uno scanner che comprendeva 6 piastrine con 10 semi ciascuna, vengono segmentati i singoli semi ed associati ad un dataset la cui label di germinazione è stata fatta manualmente.
Si vuole provare a automatizzare questo processo per sapere quando un seme è germinarto e indaghiamo un approccio con varie CNN.

## 📁 Project structure

```text
.
├── project/
│   ├── config.py              # Configurazione (path, iperparametri, architetture)
│   ├── data_processing.py     # CSV cleaning, labeling immagini, split by seed, Dataset PyTorch
│   ├── model.py               # build_model(): custom + modelli pre-trained (torchvision)
│   ├── main.py                # training + valutazione + report + salvataggio modello
│   └── utils.py               # plot_history(), pick_threshold(), ecc.
│
├── plots/                     # grafici metriche (.png)
├── reports/                   # report di valutazione (.md)
└── segmentation.ipynb         # segmentazione singoli semi e creazione dataset
```
