import io
from unittest.mock import patch

import pytest
import torch
from PIL import Image


def make_test_image_bytes():
    img = Image.new("RGB", (64, 64), color=(0, 255, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class DummyModel:
    def __init__(self):
        self.tda_feature_dim = 13

    def predict_with_uncertainty(self, img_tensor, tda_tensor, n_forward=10):
        return torch.tensor([[0.1, 0.9]]), torch.tensor([[0.05]]), None


class DummyExplainer:
    def explain(self, img_tensor, tda_tensor):
        return {
            "predicted_class": 1,
            "class_probs": [[0.1, 0.9]],
            "gradcam": [torch.zeros((1, 224, 224)).numpy()],
            "gradcam_pp": [torch.zeros((1, 224, 224)).numpy()],
            "int_grads": [torch.zeros((1, 224, 224)).numpy()],
            "guided_bp": [torch.zeros((1, 224, 224)).numpy()],
            "tda_importance": [0.1] * 13,
        }


def test_explain_success(client):
    image_bytes = make_test_image_bytes()
    dummy_tensor = torch.ones((1, 3, 224, 224), dtype=torch.float32)
    dummy_tda = torch.ones((1, 13), dtype=torch.float32)

    with patch("backend.app.routers.inference.get_model", return_value=DummyModel()), \
         patch("backend.app.routers.inference.get_transform"), \
         patch("backend.app.routers.inference.get_explainer", return_value=DummyExplainer()), \
         patch("backend.app.routers.inference.get_class_names", return_value=["A", "B"]), \
         patch("backend.app.routers.inference.xai_service.prepare_inputs", return_value=(dummy_tensor, dummy_tda, None)), \
         patch("backend.app.routers.inference.xai_service.render_heatmap_b64", return_value="data:image/png;base64,aGVsbG8="), \
         patch("backend.app.routers.inference.xai_service.render_comparison_b64", return_value="data:image/png;base64,Y29tcA=="), \
         patch("backend.app.routers.inference.xai_service.render_tda_b64", return_value="data:image/png;base64,dGRh"), \
         patch("backend.app.routers.inference.xai_service.render_confidence_b64", return_value="data:image/png;base64,Y29uZg=="), \
         patch("backend.app.routers.inference.xai_service.audit_log"):
        response = client.post(
            "/explain",
            files={"file": ("test.jpg", image_bytes, "image/jpeg")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["predicted_class"] == "B"
    assert payload["predicted_index"] == 1
    assert payload["confidence"] == pytest.approx(0.9)
    assert payload["heatmap_gradcam"].startswith("data:image/png;base64")
    assert payload["chart_tda"].startswith("data:image/png;base64")
    assert payload["clinical_reasoning"] is not None


def test_explain_invalid_file(client):
    with patch("backend.app.routers.inference.get_model", return_value=DummyModel()), \
         patch("backend.app.routers.inference.get_transform"), \
         patch("backend.app.routers.inference.xai_service.prepare_inputs", side_effect=ValueError("bad image")):
        response = client.post(
            "/explain",
            files={"file": ("test.txt", b"not-an-image", "text/plain")},
        )

    assert response.status_code == 422
    assert "Image loading error" in response.json()["detail"]
