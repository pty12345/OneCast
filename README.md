# OneCast
OneCast: Structured Decomposition and Modular Generation for Cross-Domain Time Series Forecasting

### Prepare environments
```bash
pip install -r requirements.txt
```

### Prepare datasets

```bash
### Download CzeLan, FRED-MD, NYSE, Covid-19, Wike2000 from Time Series Benchmark
# TFB: Towards Comprehensive and Fair Benchmarking of Time Series Forecasting Methods

### Download ETTh2, ETTm2, electricity, weather, traffic from Autoformer
# https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy
```

### Train Time Series Tokenizer

```bash

cd TSTokenizer

# large datasets' tokenizer
bash scripts/Cross_CzeLan.sh
bash scripts/Cross_Traffic.sh
bash scripts/Cross_ETTh2_ETTm2_weather.sh

# small datasets' tokenizer
bash scripts/Cross_Wike2000.sh
bash scripts/Cross_FRED_Covid_NYSE.sh
```

### Train Unify Model
```bash
bash scripts/unify/Cross_large_unify.sh
# or 
bash scripts/unify/Cross_small_unify.sh
```

### Train Adaptive Model
```bash
bash scripts/adaptive/Cross_large_unify.sh
# or 
bash scripts/adaptive/Cross_small_unify.sh
```

