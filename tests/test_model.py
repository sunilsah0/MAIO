import pytest
from app.model import DiabetesModel

def test_predict_output_type(tmp_path, monkeypatch):
    # Create a dummy model to inject
    class Dummy:
        def predict(self, x):
            return [123.45]
    dummy_model = Dummy()
    monkeypatch.setattr("app.model.joblib.load", lambda path: dummy_model)
    from app.model import DiabetesModel
    dm = DiabetesModel(model_path="dummy_path")
    feat = {k: 0.0 for k in ["age","sex","bmi","bp","s1","s2","s3","s4","s5","s6"]}
    pred = dm.predict(feat)
    assert isinstance(pred, float)
    assert pytest.approx(pred, 0.1) == 123.45
