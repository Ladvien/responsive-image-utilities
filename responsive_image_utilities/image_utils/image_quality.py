from __future__ import annotations  # Needed for type hinting
from dataclasses import dataclass, asdict
from PIL import Image as PILImage
import torch
import piq


from responsive_image_utilities.image_noiser import ImageNoiser
from responsive_image_utilities.utils import map_value, preprocess_image


@dataclass
class BrisqueScore:
    original_image_score: float
    corrupted_image_score: float
    new_image_score: float
    severity_used: float
    normalized_score: float


@dataclass
class ClipIQAScore:
    original_image_score: float
    corrupted_image_score: float
    new_image_score: float
    severity_used: float
    normalized_score: float


@dataclass
class ImageQualityScores:
    brisque: BrisqueScore
    clip_iqa: ClipIQAScore


class ImageQualityAssessor:

    def __init__(self, image: PILImage.Image, baseline_severity: float = 0.2):
        self.image = image
        self.baseline_severity = baseline_severity

    def normalized_score(
        self,
        original_image: PILImage.Image,
        corrupted_image: PILImage.Image,
        new_image: PILImage.Image,
    ) -> ImageQualityScores:

        # 1. Created a corrupted image
        # 2. Calculate the image quality of the corrupted image
        # 3. Calculate the image quality of the original image
        # 4. Calculate the image quality of the new image.
        # 5. Normalize the new image quality score by the original
        #    image quality score and the corrupted image quality score.

        resized_new_image = new_image.resize(
            (original_image.size[0], original_image.size[1])
        )

        original_brisque_score = self.calculate_brisque(original_image)
        corrupted_brisque_score = self.calculate_brisque(corrupted_image)
        new_image_brisque_score = self.calculate_brisque(resized_new_image)

        original_clip_iqa_score = self.calculate_clip_iqa(original_image)
        corrupted_clip_iqa_score = self.calculate_clip_iqa(corrupted_image)
        new_image_clip_iqa_score = self.calculate_clip_iqa(resized_new_image)

        normalized_brisque_score = map_value(
            new_image_brisque_score,
            corrupted_brisque_score,
            original_brisque_score,
            0,
            1,
        )

        normalized_clip_iqa_score = map_value(
            new_image_clip_iqa_score,
            corrupted_clip_iqa_score,
            original_clip_iqa_score,
            0,
            1,
        )

        brisque_score = BrisqueScore(
            original_image_score=original_brisque_score,
            corrupted_image_score=corrupted_brisque_score,
            new_image_score=new_image_brisque_score,
            severity_used=self.baseline_severity,
            normalized_score=normalized_brisque_score,
        )

        clip_iqa_score = ClipIQAScore(
            original_image_score=original_clip_iqa_score,
            corrupted_image_score=corrupted_clip_iqa_score,
            new_image_score=new_image_clip_iqa_score,
            severity_used=self.baseline_severity,
            normalized_score=normalized_clip_iqa_score,
        )

        return ImageQualityScores(
            brisque=brisque_score,
            clip_iqa=clip_iqa_score,
        )

    def calculate_clip_iqa(self, image: PILImage.Image) -> float:
        image_tensor = preprocess_image(image)
        max_value = image_tensor.max().item()
        if max_value > 1.0:
            image_tensor = image_tensor / 255.0

        image_clip_iqa_index = piq.CLIPIQA(data_range=1.0).to(image_tensor)(
            image_tensor.clone().permute(2, 0, 1)[None, ...]
        )[0][0]

        return image_clip_iqa_index.item()

    def calculate_brisque(self, image: PILImage.Image) -> float:
        image_tensor = preprocess_image(image)
        image_tensor = image_tensor.permute(2, 0, 1)[None, ...] / 255.0
        brisque_index: torch.Tensor = piq.brisque(
            image_tensor, data_range=1.0, reduction="none"
        )
        return brisque_index.item()

    def __dict__(self) -> dict[str, float]:
        return asdict(self)
