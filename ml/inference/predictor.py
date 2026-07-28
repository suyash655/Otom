"""
Inference module for HybridModel predictions.

Provides clean separation between training and inference logic.
"""
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import importlib.util

# Load dependencies from ML module structure
here = Path(__file__).parent.parent


def _load_module(name: str, rel_path: str):
    """Load a Python module from a relative path."""
    module_path = here / rel_path
    spec = importlib.util.spec_from_file_location(name, str(module_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Load model and dataset modules
hybrid_mod = _load_module("hybrid_model", "models/hybrid_model.py")
dataset_mod = _load_module("dataset", "preprocessing/dataset.py")

HybridModel = hybrid_mod.HybridModel
CLASS_NAMES = dataset_mod.CLASS_NAMES
CLASS_TO_IDX = dataset_mod.CLASS_TO_IDX
get_eval_transforms = dataset_mod.get_eval_transforms


class ModelPredictor:
    """Wrapper for model inference with proper device handling."""
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to .pth checkpoint file
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = torch.device(device)
        self.checkpoint_path = Path(checkpoint_path)
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Initialize model
        num_classes = checkpoint.get("num_classes", len(CLASS_NAMES))
        tda_dim = 13  # Standard TDA feature dimension
        
        self.model = HybridModel(
            num_classes=num_classes,
            tda_feature_dim=tda_dim,
            pretrained=False,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        
        # Store metadata
        self.class_names = checkpoint.get("class_names", CLASS_NAMES)
        self.num_classes = num_classes
        
        # Setup transforms
        self.transform = get_eval_transforms()
        
    def predict(
        self,
        image: torch.Tensor,
        tda_features: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Run single image prediction.
        
        Args:
            image: Preprocessed tensor (C, H, W) or batch (B, C, H, W)
            tda_features: Optional TDA feature tensor (B, 13) or (13,)
            
        Returns:
            Dictionary with prediction results
        """
        # Ensure batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        # Handle TDA features
        if tda_features is None:
            tda_features = torch.zeros(image.shape[0], 13, device=self.device)
        elif tda_features.dim() == 1:
            tda_features = tda_features.unsqueeze(0)
        
        tda_features = tda_features.to(self.device)
        image = image.to(self.device)
        
        # Run inference
        with torch.no_grad():
            logits = self.model(image, tda_features)
            probs = torch.softmax(logits, dim=1)
            confidence, predicted = probs.max(dim=1)
        
        # Convert to numpy for easier handling
        predicted = predicted.cpu().numpy()
        confidence = confidence.cpu().numpy()
        probs = probs.cpu().numpy()
        
        # Build result
        result = {
            "predicted_class": self.class_names[predicted[0]],
            "predicted_index": int(predicted[0]),
            "confidence": float(confidence[0]),
            "class_probs": {
                name: float(probs[0][i]) 
                for i, name in enumerate(self.class_names)
            },
        }
        
        return result
    
    def predict_batch(
        self,
        images: torch.Tensor,
        tda_features: Optional[torch.Tensor] = None,
    ) -> List[Dict]:
        """
        Run batch prediction.
        
        Args:
            images: Batch of preprocessed tensors (B, C, H, W)
            tda_features: Optional TDA features (B, 13)
            
        Returns:
            List of prediction dictionaries
        """
        if tda_features is None:
            tda_features = torch.zeros(images.shape[0], 13, device=self.device)
        
        tda_features = tda_features.to(self.device)
        images = images.to(self.device)
        
        with torch.no_grad():
            logits = self.model(images, tda_features)
            probs = torch.softmax(logits, dim=1)
            confidence, predicted = probs.max(dim=1)
        
        predicted = predicted.cpu().numpy()
        confidence = confidence.cpu().numpy()
        probs = probs.cpu().numpy()
        
        results = []
        for i in range(images.shape[0]):
            results.append({
                "predicted_class": self.class_names[predicted[i]],
                "predicted_index": int(predicted[i]),
                "confidence": float(confidence[i]),
                "class_probs": {
                    name: float(probs[i][j]) 
                    for j, name in enumerate(self.class_names)
                },
            })
        
        return results
    
    def predict_with_uncertainty(
        self,
        image: torch.Tensor,
        tda_features: Optional[torch.Tensor] = None,
        mc_samples: int = 30,
    ) -> Dict:
        """
        Run prediction with Monte Carlo dropout uncertainty estimation.
        
        Args:
            image: Preprocessed tensor (C, H, W)
            tda_features: Optional TDA features (13,)
            mc_samples: Number of MC dropout samples
            
        Returns:
            Dictionary with prediction and uncertainty metrics
        """
        # Enable dropout during inference
        self.model.train()
        
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        if tda_features is None:
            tda_features = torch.zeros(image.shape[0], 13, device=self.device)
        elif tda_features.dim() == 1:
            tda_features = tda_features.unsqueeze(0)
        
        tda_features = tda_features.to(self.device)
        image = image.to(self.device)
        
        # Collect MC samples
        all_probs = []
        with torch.no_grad():
            for _ in range(mc_samples):
                logits = self.model(image, tda_features)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())
        
        # Disable dropout after inference
        self.model.eval()
        
        # Calculate statistics
        all_probs = np.array(all_probs)  # (mc_samples, B, num_classes)
        mean_probs = all_probs.mean(axis=0)[0]
        std_probs = all_probs.std(axis=0)[0]
        
        # Prediction based on mean probabilities
        predicted_idx = mean_probs.argmax()
        confidence = mean_probs[predicted_idx]
        
        # Uncertainty as standard deviation of predicted class
        uncertainty = std_probs[predicted_idx]
        
        return {
            "predicted_class": self.class_names[predicted_idx],
            "predicted_index": int(predicted_idx),
            "confidence": float(confidence),
            "uncertainty": float(uncertainty),
            "class_probs": {
                name: float(mean_probs[i]) 
                for i, name in enumerate(self.class_names)
            },
            "low_confidence": confidence < 0.5,
            "uncertain": uncertainty > 0.15,
        }


def load_predictor(checkpoint_path: str, device: str = "cpu") -> ModelPredictor:
    """
    Convenience function to load a predictor.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to use for inference
        
    Returns:
        Initialized ModelPredictor instance
    """
    return ModelPredictor(checkpoint_path, device=device)