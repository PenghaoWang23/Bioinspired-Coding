from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from .codebook import CodebookNode, build_codebook_trees, build_pixel_lookup
from .codec import build_inner_rs_positions, encode_inner_rs, int_to_dna_address


@dataclass(frozen=True)
class EncodedSegments:
    protected_segments: List[str]
    original_pixel_count: int
    padded_pixel_count: int
    padding_pixels: int
    payload_length: int
    codeword_length: int
    padding_value: int
    address_length: int
    segment_count: int


def create_codebook(
    pixel_values_by_tree: Dict[int, Iterable[int]],
    depth: int = 5,
) -> List[CodebookNode]:
    from .codebook import assign_pixels_to_trees

    trees = build_codebook_trees(depth=depth)
    assign_pixels_to_trees(trees, pixel_values_by_tree)
    return trees


def encode_pixels_to_dna(trees: Sequence[CodebookNode], pixel_values: Iterable[int]) -> str:
    pixel_lookup = build_pixel_lookup(trees)
    dna_sequence_parts: List[str] = []

    for pixel in pixel_values:
        pixel_value = int(pixel)
        try:
            dna_sequence_parts.append(pixel_lookup[pixel_value])
        except KeyError as exc:
            raise ValueError(f"No DNA codeword found for pixel value {pixel_value}.") from exc
    return "".join(dna_sequence_parts)


def pad_pixel_values(
    pixel_values: Iterable[int],
    pixels_per_segment: int,
    padding_value: int = 0,
) -> tuple[List[int], int]:
    pixel_list = [int(pixel) for pixel in pixel_values]
    if pixels_per_segment <= 0:
        raise ValueError("pixels_per_segment must be positive.")

    remainder = len(pixel_list) % pixels_per_segment
    padding_pixels = 0 if remainder == 0 else pixels_per_segment - remainder
    if padding_pixels:
        pixel_list.extend([padding_value] * padding_pixels)
    return pixel_list, padding_pixels


def split_dna_sequence(dna_sequence: str, payload_length: int = 120) -> List[str]:
    return [
        dna_sequence[index:index + payload_length]
        for index in range(0, len(dna_sequence), payload_length)
        if dna_sequence[index:index + payload_length]
    ]


def add_addresses_to_segments(segments: Sequence[str], address_length: int = 8) -> List[str]:
    return [
        f"{int_to_dna_address(index, address_length=address_length)}{segment}"
        for index, segment in enumerate(segments)
    ]


def encode_segments_with_inner_rs(
    dna_sequence: str,
    original_pixel_count: int,
    payload_length: int = 120,
    codeword_length: int = 5,
    check_size: int = 4,
    padding_value: int = 0,
    address_length: int = 8,
    positions: Optional[Sequence[int]] = None,
) -> EncodedSegments:
    protected_payload_segments: List[str] = []
    padded_pixel_count = len(dna_sequence) // codeword_length
    if payload_length % codeword_length != 0:
        raise ValueError("payload_length must be divisible by codeword_length.")
    rs_positions = list(positions) if positions is not None else build_inner_rs_positions(payload_length)

    for segment in split_dna_sequence(dna_sequence, payload_length=payload_length):
        if len(segment) != payload_length:
            raise ValueError("Each payload segment must match payload_length before RS encoding.")
        protected_payload_segments.append(encode_inner_rs(segment, rs_positions, check_size))
    addressed_segments = add_addresses_to_segments(protected_payload_segments, address_length=address_length)
    return EncodedSegments(
        protected_segments=addressed_segments,
        original_pixel_count=original_pixel_count,
        padded_pixel_count=padded_pixel_count,
        padding_pixels=padded_pixel_count - original_pixel_count,
        payload_length=payload_length,
        codeword_length=codeword_length,
        padding_value=padding_value,
        address_length=address_length,
        segment_count=len(addressed_segments),
    )
