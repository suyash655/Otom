import io
from unittest.mock import patch

import pytest
import torch
from PIL import Image


def make_test_image_bytes():
    img = Image.new("RGB", (64, 64), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class DummyModel:
    def __init__(self):
        self.tda_feature_dim = 13

    def __call__(self, img_tensor, tda_tensor):
        return torch.tensor([[0.1, 0.9]], dtype=torch.float32)

    def predict_with_uncertainty(self, img_tensor, tda_tensor, n_forward=10):
        return torch.tensor([[0.1, 0.9]]), torch.tensor([[0.05]]), None


@pytest.mark.parametrize("expected_status", [200])
def test_predict_success(client, expected_status):
    image_bytes = make_test_image_bytes()
    dummy_tensor = torch.ones((1, 3, 224, 224), dtype=torch.float32)
    dummy_tda = torch.ones((1, 13), dtype=torch.float32)

    with patch("backend.app.routers.inference.get_model", return_value=DummyModel()), \
         patch("backend.app.routers.inference.get_transform"), \
         patch("backend.app.routers.inference.get_class_names", return_value=["A", "B"]), \
         patch("backend.app.routers.inference.xai_service.prepare_inputs", return_value=(dummy_tensor, dummy_tda, None)), \
         patch("backend.app.routers.inference.xai_service.audit_log"):
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", image_bytes, "image/jpeg")},
        )

    assert response.status_code == expected_status
    payload = response.json()
    assert payload["predicted_class"] == "B"
    assert payload["predicted_index"] == 1
    assert payload["confidence"] == pytest.approx(0.9)
    assert payload["class_probs"]["A"] == pytest.approx(0.1)
    assert payload["class_probs"]["B"] == pytest.approx(0.9)
    assert payload["low_confidence"] is False
    assert payload["uncertain"] is False


def test_predict_invalid_file(client):
    with patch("backend.app.routers.inference.get_model", return_value=DummyModel()), \
         patch("backend.app.routers.inference.get_transform"), \
         patch("backend.app.routers.inference.xai_service.prepare_inputs", side_effect=ValueError("bad image")):
        response = client.post(
            "/predict",
            files={"file": ("test.txt", b"not-an-image", "text/plain")},
        )

    assert response.status_code == 422
    assert "Image loading error" in response.json()["detail"]


def test_predict_internal_error(client):
    image_bytes = make_test_image_bytes()
    dummy_tensor = torch.ones((1, 3, 224, 224), dtype=torch.float32)
    dummy_tda = torch.ones((1, 13), dtype=torch.float32)

    with patch("backend.app.routers.inference.get_model", return_value=DummyModel()), \
         patch("backend.app.routers.inference.get_transform"), \
         patch("backend.app.routers.inference.xai_service.prepare_inputs", return_value=(dummy_tensor, dummy_tda, None)), \
         patch.object(DummyModel, "predict_with_uncertainty", side_effect=RuntimeError("failed")), \
         patch("backend.app.routers.inference.xai_service.audit_log"):
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", image_bytes, "image/jpeg")},
        )

    assert response.status_code == 500
    assert "Prediction error" in response.json()["detail"]
