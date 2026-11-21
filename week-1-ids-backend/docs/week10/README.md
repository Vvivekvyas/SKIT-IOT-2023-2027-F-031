# Week 10 (3-9 Nov)

FT-Transformer data preparation, secure model-service interface, Transformer input/feature analysis, model-input/API tests.

| Path | Kaam |
|---|---|
| `ml/ft_transformer/data_prep.py` | FT-Transformer data preparation |
| `ml/ft_transformer/feature_analysis.py` | Transformer input / feature analysis |
| `ml/ft_transformer/model.py` | FT-Transformer model + train loop |
| `ml/ft_transformer/service.py` | secure model-service interface |
| `scripts/run_week10_ft_transformer.py` | prep, analysis, training ek saath |
| `tests/test_week10_model_input_api.py` | model-input / API tests |
| `ml/artifacts/ft_transformer/` | saved prep + model.pt (script banati hai) |
| `docs/week10/` | analysis reports (script banati hai) |

## Run

```bash
pip install -r requirements-week10.txt
python scripts/run_week10_ft_transformer.py --data data/raw/CIC-IoT2023 --target label
pytest tests/test_week10_model_input_api.py -q
FT_API_KEY=your-secret uvicorn ml.ft_transformer.service:app_factory --factory --port 8001
```
