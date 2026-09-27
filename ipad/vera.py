"""Paper-first VERA inference mathematics; independent of labels and VLM libraries."""
from dataclasses import asdict, dataclass
import hashlib
import json
import re

import numpy as np


QUESTIONS = "\n".join([
    "1. Are there any people in the video who are not in their typical positions or engaging in activities that are not consistent with their usual behavior?",
    "2. Are there any vehicles in the video that are not in their typical positions or being used in a way that is not consistent with their usual function?",
    "3. Are there any objects in the video that are not in their typical positions or being used in a way that is not consistent with their usual function?",
    "4. Is there any visible damage or unusual movement in the video that indicates an anomaly?",
    "5. Are there any unusual sounds or noises in the video that suggest an anomaly?",
]) + "\n"


@dataclass(frozen=True)
class VeraConfig:
    seed: int = 0
    fps_assumption: int = 30
    center_stride: int = 16
    window_seconds: int = 10
    sampled_frames: int = 8
    retrieval_fraction: float = .1
    temperature: float = 10.
    kernel_size: int = 15
    sigma1: float = 10.
    model_id: str = "OpenGVLab/InternVL2-8B"
    model_revision: str = "6fb9ad6924f69424e57fab2ab061d707688f0296"
    vera_commit: str = "15b8bcb8574a977c229c577f50bfe6f06d07106e"
    imagebind_commit: str = "53680b02d7e37b19b124fa37bae4b6c98c38f5be"
    lavad_commit: str = "1ad46c666d1b3cfb262f3dd84769acf873285056"
    imagebind_samples: int = 10
    imagebind_precision: str = "float32"
    vlm_precision: str = "bfloat16"
    attention: str = "eager"
    max_new_tokens: int = 1024

    def digest(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def segments(length, config=VeraConfig()):
    if length < 1:
        raise ValueError("Empty video")
    half = config.fps_assumption * config.window_seconds // 2
    result = []
    for center in range(0, length, config.center_stride):
        start, end = max(0, center-half), min(length, center+half)
        ids = (start + np.arange(config.sampled_frames) * (end-start) / config.sampled_frames).astype(int)
        result.append(dict(center=center, start=start, end=end, frame_ids=ids.tolist(),
                           score_end=min(length, center+config.center_stride)))
    return result


def build_prompt(template):
    marker = "Based on the analysis above"
    if template.count(marker) != 1 or template.count("$Data") != 1:
        raise ValueError("Unexpected upstream learner template")
    prefix = ''.join(f'Frame{i+1}: <image>\n' for i in range(8))
    return template.replace(marker, QUESTIONS + marker).replace('$Data', prefix)


def parse_response(response):
    # Never classify malformed or truncated prose as a positive/negative prediction.
    matches = re.findall(r"(?:^|\n)\s*(?:\*\*)?Output(?:\*\*)?\s*:\s*(?:\*\*)?\s*([01])\s*(?:```)?\s*$", response)
    if len(matches) != 1:
        raise ValueError("Missing or ambiguous terminal Output: 0/1")
    return int(matches[0])


def refine_scores(initial, features, length, config=VeraConfig()):
    initial = np.asarray(initial, dtype=np.float64)
    features = np.asarray(features, dtype=np.float64)
    h = len(segments(length, config))
    if initial.shape != (h,) or features.ndim != 2 or features.shape[0] != h:
        raise ValueError("Unaligned segment scores/features")
    if not np.isin(initial, [0, 1]).all() or not np.isfinite(features).all():
        raise ValueError("Invalid segment data")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Zero feature vector")
    features = features / norms
    similarity = features @ features.T
    k = max(1, int(config.retrieval_fraction*h))
    neighbors = np.argsort(-similarity, axis=1, kind='stable')[:, :k]
    logits = np.take_along_axis(similarity, neighbors, axis=1) / config.temperature
    weights = np.exp(logits-logits.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    retrieved = (initial[neighbors]*weights).sum(axis=1)
    half = config.kernel_size // 2
    positions = np.arange(-half, half+1)
    kernel = np.exp(-positions**2/(2*config.sigma1**2))
    kernel /= kernel.sum()
    # Explicit zero padding; preserves h even when h < kernel_size.
    smooth = np.convolve(np.pad(retrieved, (half, half)), kernel, mode='valid')
    expanded = {name: np.repeat(values, config.center_stride)[:length]
                for name, values in [('initial', initial), ('retrieved', retrieved), ('smoothed', smooth)]}
    # Paper Eq. 5 uses one-based i and floor(F/2); sigma=1 for the undefined F=1 case.
    center = length//2
    position_weight = np.exp(-(np.arange(1, length+1)-center)**2/(2*max(1, center)**2))
    expanded['final'] = expanded['smoothed']*position_weight
    return expanded, dict(neighbors=neighbors.tolist(), weights=weights.tolist(),
                          retrieved=retrieved.tolist(), smoothed=smooth.tolist(),
                          position_weights=position_weight.tolist())
