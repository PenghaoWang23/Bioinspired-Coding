from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence

from .codebook import CodebookNode
from .decoder import DecodedPixels, decode_protected_dna_to_pixels
from .encoder import EncodedSegments, create_codebook, encode_pixels_to_dna, encode_segments_with_inner_rs, pad_pixel_values
from .codec import expected_segment_count

if TYPE_CHECKING:
    import torch

    from .transformer_model import TransformerDNA


@dataclass(frozen=True)
class WorkflowConfig:
    tree_depth: int = 5
    payload_length: int = 120
    inner_rs_check_size: int = 4
    codeword_length: int = 5
    padding_value: int = 0
    address_length: int = 8


def build_codebook_from_pixels(
    pixel_values_by_tree: Dict[int, Iterable[int]],
    config: WorkflowConfig = WorkflowConfig(),
) -> List[CodebookNode]:
    return create_codebook(pixel_values_by_tree=pixel_values_by_tree, depth=config.tree_depth)


def encode_pixels(
    pixel_values: Iterable[int],
    trees: Sequence[CodebookNode],
    config: WorkflowConfig = WorkflowConfig(),
) -> EncodedSegments:
    pixels_per_segment = config.payload_length // config.codeword_length
    padded_pixels, padding_pixels = pad_pixel_values(
        pixel_values=pixel_values,
        pixels_per_segment=pixels_per_segment,
        padding_value=config.padding_value,
    )
    dna_sequence = encode_pixels_to_dna(trees=trees, pixel_values=padded_pixels)
    return encode_segments_with_inner_rs(
        dna_sequence=dna_sequence,
        original_pixel_count=len(padded_pixels) - padding_pixels,
        payload_length=config.payload_length,
        codeword_length=config.codeword_length,
        check_size=config.inner_rs_check_size,
        padding_value=config.padding_value,
        address_length=config.address_length,
    )


def decode_segments(
    protected_segments: Sequence[str],
    trees: Sequence[CodebookNode],
    config: WorkflowConfig = WorkflowConfig(),
    original_pixel_count: Optional[int] = None,
) -> DecodedPixels:
    expected_count = None
    if original_pixel_count is not None:
        expected_count = expected_segment_count(
            pixel_count=original_pixel_count,
            payload_length=config.payload_length,
            codeword_length=config.codeword_length,
        )
    return decode_protected_dna_to_pixels(
        segments=protected_segments,
        trees=trees,
        check_size=config.inner_rs_check_size,
        payload_length=config.payload_length,
        original_pixel_count=original_pixel_count,
        codeword_length=config.codeword_length,
        address_length=config.address_length,
        expected_segment_count=expected_count,
    )


def reconstruct_segment_with_transformer(
    model: "TransformerDNA",
    noisy_copies: Sequence[str],
    device: "torch.device | str",
) -> str:
    from .transformer_model import reconstruct_sequence

    return reconstruct_sequence(model=model, copies=noisy_copies, device=device)


def reconstruct_segments_with_transformer(
    model: "TransformerDNA",
    segment_copies: Sequence[Sequence[str]],
    device: "torch.device | str",
) -> List[str]:
    return [
        reconstruct_segment_with_transformer(model=model, noisy_copies=copies, device=device)
        for copies in segment_copies
    ]


def decode_from_multiple_copies(
    segment_copies: Sequence[Sequence[str]],
    trees: Sequence[CodebookNode],
    config: WorkflowConfig = WorkflowConfig(),
    original_pixel_count: Optional[int] = None,
    model: "TransformerDNA" = None,
    device: "torch.device | str" = None,
) -> DecodedPixels:
    if not segment_copies:
        return DecodedPixels(
            pixels=[],
            decoded_segments=[],
            padded_pixel_count=0,
            trimmed_pixel_count=0,
            padding_pixels_removed=0,
            ordered_segments=[],
        )
    if model is None or device is None:
        raise ValueError("decode_from_multiple_copies requires both a transformer model and a device.")
    protected_segments = reconstruct_segments_with_transformer(model=model, segment_copies=segment_copies, device=device)

    return decode_segments(
        protected_segments=protected_segments,
        trees=trees,
        config=config,
        original_pixel_count=original_pixel_count,
    )
