# AI-powered-seismic-signal-enhancement-using-deep-ensemble-learning
A hybrid deep learning framework integrating DnCNN, U-Net, Noise2Noise, and MLP denoisers with Continuous Wavelet Transform and CNN-based classification for intelligent seismic noise reduction, signal reconstruction, and event detection. The hybrid approach uses adaptive feature weighting and fusion to enhance seismic analysis performance.
Software used- Google colab.
- `training_code.py` – trains all denoising models (DnCNN, U-Net, N2N, MLP, Classical, Ensemble). add the sample file while running the code.
- `testing_code.py` – loads trained models, runs inference, computes metrics, and generates all plots (denoised signals, SNR, residual error, scalograms).
- `sample.csv` – required input file containing seismic traces (`trace_data`, `trace_category`) used by both training and testing scripts.
