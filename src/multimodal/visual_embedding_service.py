"""Visual embedding service for cross-modal image and text semantic representations.

Uses OpenAI CLIP (clip-vit-base-patch32, 512 dimensions) to map both cybersecurity
visual artifacts (screenshots, diagrams, tables) and natural-language investigator
queries into a shared cross-modal embedding space with L2 normalization.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class BaseVisualEmbeddingService(ABC):
    """Abstract base class for visual embedding models."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding dimension."""
        pass

    @abstractmethod
    def embed_images(self, image_paths: list[str | Path]) -> np.ndarray:
        """Generate normalized visual embeddings for a list of image paths."""
        pass

    @abstractmethod
    def embed_text_query(self, query: str) -> np.ndarray:
        """Generate normalized cross-modal text embedding in the shared visual space."""
        pass


class CLIPVisualEmbeddingService(BaseVisualEmbeddingService):
    """CLIP-based visual embedding service supporting GPU and CPU inference."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str | None = None,
    ) -> None:
        """Initialize CLIP visual embedding service.

        Args:
            model_name: HuggingFace model identifier.
            device: 'cuda', 'cpu', or None (auto-detect).
        """
        self.model_name = model_name
        self._model: Any = None
        self._processor: Any = None
        self._device_str = device
        self._dimension = 512
        logger.info("CLIPVisualEmbeddingService configured with model: %s", model_name)

    def _load_model(self) -> None:
        """Lazy-load the CLIP model and processor."""
        if self._model is not None:
            return

        import torch
        from transformers import CLIPModel, CLIPProcessor

        if self._device_str is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self._device_str

        logger.info("Loading CLIP model %s on device '%s' ...", self.model_name, self.device)
        self._processor = CLIPProcessor.from_pretrained(self.model_name)
        self._model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self._model.eval()
        self._dimension = self._model.projection_dim
        logger.info("CLIP model loaded successfully (dimension: %d, device: %s)", self._dimension, self.device)

    @property
    def dimension(self) -> int:
        """Return embedding dimension (512 for ViT-B/32)."""
        self._load_model()
        return self._dimension

    def embed_images(
        self,
        image_paths: list[str | Path],
        batch_size: int = 16,
    ) -> np.ndarray:
        """Embed a list of images into L2-normalized 512-dim vectors.

        Args:
            image_paths: List of file paths to images.
            batch_size: Batch size for encoding.

        Returns:
            NumPy array of shape (N, 512) with float32 normalized vectors.
        """
        if not image_paths:
            return np.array([], dtype=np.float32).reshape(0, self.dimension)

        self._load_model()
        import torch

        all_embeddings: list[np.ndarray] = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            pil_images = []
            for p in batch_paths:
                img_path = Path(p)
                if not img_path.exists():
                    raise FileNotFoundError(f"Image not found for embedding: {img_path}")
                pil_images.append(Image.open(str(img_path)).convert("RGB"))

            with torch.no_grad():
                inputs = self._processor(images=pil_images, return_tensors="pt").to(self.device)
                outputs = self._model.get_image_features(**inputs)
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    image_features = outputs.pooler_output
                elif hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
                    image_features = outputs.image_embeds
                elif isinstance(outputs, torch.Tensor):
                    image_features = outputs
                else:
                    image_features = outputs[0]

                # Normalize features to unit sphere (L2 norm) for cosine inner product
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                embeddings = image_features.cpu().numpy().astype(np.float32)
                all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)

    def embed_image(self, image_path: str | Path) -> np.ndarray:
        """Embed a single image into a (1, 512) vector."""
        return self.embed_images([image_path])

    def embed_text_query(self, query: str) -> np.ndarray:
        """Embed natural language query into the shared cross-modal visual embedding space.

        Args:
            query: Text query string.

        Returns:
            NumPy array of shape (1, 512) with float32 normalized vector.
        """
        self._load_model()
        import torch

        with torch.no_grad():
            inputs = self._processor(text=[query], return_tensors="pt", padding=True, truncation=True).to(self.device)
            outputs = self._model.get_text_features(**inputs)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                text_features = outputs.pooler_output
            elif hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
                text_features = outputs.text_embeds
            elif isinstance(outputs, torch.Tensor):
                text_features = outputs
            else:
                text_features = outputs[0]

            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features.cpu().numpy().astype(np.float32)
