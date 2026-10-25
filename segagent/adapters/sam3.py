"""Segmenter adapter: SAM 3 via Hugging Face transformers. Encodes each image once, then runs text prompts."""

import threading

import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

from segagent.core.domain import Detection

MODEL_ID = "facebook/sam3"


class Sam3:
    def __init__(self, model_id: str = MODEL_ID):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = Sam3Model.from_pretrained(model_id, dtype=dtype).to(self.device).eval()
        self.processor = Sam3Processor.from_pretrained(model_id)
        self._lock = threading.Lock()  # one GPU: concurrent runs take turns
        # Cache features for the last image *object*. Holding the reference means a new image can never be
        # mistaken for it, unlike a file name or id().
        self._image = None
        self._vision = None

    def _vision_embeds(self, image: Image.Image):
        if image is not self._image:
            self._vision = None  # free GPU memory before encoding the next image
            inputs = self.processor(images=image, return_tensors="pt").to(self.device, dtype=self.model.dtype)
            with torch.inference_mode():
                self._vision = self.model.get_vision_features(pixel_values=inputs.pixel_values)
            self._image = image
        return self._vision

    def _run(self, image: Image.Image, phrase: str, threshold: float, **prompt) -> tuple[list[Detection], float]:
        """One prompt against the cached image features. Caller holds the lock."""
        inputs = self.processor(**prompt, return_tensors="pt")
        inputs.pop("original_sizes", None)
        inputs = {k: v.to(self.device, dtype=self.model.dtype) if v.is_floating_point() else v.to(self.device)
                  for k, v in inputs.items()}  # boxes follow the model dtype; token ids stay integers
        with torch.inference_mode():
            out = self.model(vision_embeds=self._vision_embeds(image), **inputs)
            presence = out.presence_logits.float().sigmoid().item()
            result = self.processor.post_process_instance_segmentation(
                out, threshold=threshold, mask_threshold=0.5, target_sizes=[(image.height, image.width)]
            )[0]
        return [Detection(phrase, float(s), [float(v) for v in b], m.cpu().numpy().astype(bool))
                for m, b, s in zip(result["masks"], result["boxes"], result["scores"])], presence

    def segment(self, image: Image.Image, phrases: list[str], threshold: float = 0.5):
        with self._lock:
            detections, presence = [], {}
            for phrase in phrases:
                dets, presence[phrase] = self._run(image, phrase, threshold, text=phrase)
                detections += dets
            return detections, presence

    def segment_examples(self, image: Image.Image, boxes: list[list[float]], labels: list[int], phrase: str,
                         threshold: float = 0.5):
        with self._lock:
            return self._run(image, phrase, threshold, input_boxes=[boxes], input_boxes_labels=[labels],
                             original_sizes=[[image.height, image.width]])
