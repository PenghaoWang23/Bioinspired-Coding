from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from .codebook import CodebookNode
from .codec import build_inner_rs_positions, decode_inner_rs, dna_address_to_int


@dataclass(frozen=True)
class DecodedPixels:
    pixels: List[int]
    decoded_segments: List[str]
    padded_pixel_count: int
    trimmed_pixel_count: int
    padding_pixels_removed: int
    ordered_segments: List[str]


def _first_leaf_pixel(node: CodebookNode) -> Optional[int]:
    if node.is_leaf and node.pixel_values:
        return int(node.pixel_values[0])
    for child in node.children:
        if child is None:
            continue
        pixel_value = _first_leaf_pixel(child)
        if pixel_value is not None:
            return pixel_value
    return None


def _traverse_tree(
    root: CodebookNode,
    dna_suffix: str,
    stack: Optional[List[CodebookNode]] = None,
    visited_siblings: Optional[set[int]] = None,
) -> Optional[int]:
    if not dna_suffix:
        return _first_leaf_pixel(root)

    current_stack = list(stack or [])
    current_stack.append(root)
    checked_siblings = visited_siblings or set()

    base = dna_suffix[0]
    remaining = dna_suffix[1:]
    for child in root.children:
        if child is not None and child.value == base:
            if not remaining:
                return _first_leaf_pixel(child)
            result = _traverse_tree(child, remaining, current_stack, checked_siblings)
            if result is not None:
                return result

    fallback_pixel = _first_leaf_pixel(root)
    if fallback_pixel is not None:
        return fallback_pixel

    if len(current_stack) > 1:
        parent = current_stack[-2]
        current = current_stack[-1]
        current_index = parent.children.index(current)
        for sibling_index, sibling in enumerate(parent.children):
            if sibling is None or sibling_index == current_index or sibling_index in checked_siblings:
                continue
            checked_siblings.add(sibling_index)
            result = _traverse_tree(sibling, dna_suffix, current_stack[:-1], checked_siblings)
            if result is not None:
                return result

    return None


def decode_dna_to_pixels(
    encoded_dna: str,
    trees: Sequence[CodebookNode],
    codeword_length: int = 5,
) -> List[int]:
    pixels: List[int] = []
    nucleotide_order = ["A", "T", "G", "C"]

    for index in range(0, len(encoded_dna), codeword_length):
        codeword = encoded_dna[index:index + codeword_length]
        if len(codeword) != codeword_length or codeword[0] not in nucleotide_order:
            continue
        root = trees[nucleotide_order.index(codeword[0])]
        pixel_value = _traverse_tree(root, codeword[1:])
        if pixel_value is None:
            raise ValueError(f"Unable to decode codeword: {codeword}")
        pixels.append(pixel_value)
    return pixels


def decode_segments_with_inner_rs(
    segments: Sequence[str],
    check_size: int = 4,
    payload_length: int = 120,
    positions: Optional[Sequence[int]] = None,
) -> List[str]:
    rs_positions = list(positions) if positions is not None else build_inner_rs_positions(payload_length)
    decoded_segments: List[str] = []

    for segment_index, segment in enumerate(segments):
        decoded = decode_inner_rs(
            mutated_dna=segment,
            positions=rs_positions,
            check_size=check_size,
            payload_length=payload_length,
        )
        if decoded is None:
            raise ValueError(f"RS decoding failed for segment {segment_index}.")
        decoded_segments.append(decoded)
    return decoded_segments


def split_addressed_segment(segment: str, address_length: int = 8) -> tuple[int, str]:
    if len(segment) < address_length:
        raise ValueError("Segment is shorter than the configured address length.")
    address = dna_address_to_int(segment[:address_length])
    return address, segment[address_length:]


def reorder_segments_by_address(
    segments: Sequence[str],
    expected_segment_count: int,
    address_length: int = 8,
) -> List[Optional[str]]:
    ordered_segments: List[Optional[str]] = [None] * expected_segment_count
    for segment in segments:
        address, payload = split_addressed_segment(segment, address_length=address_length)
        if 0 <= address < expected_segment_count and ordered_segments[address] is None:
            ordered_segments[address] = payload
    return ordered_segments


def fill_missing_segments_with_next(segments: Sequence[Optional[str]]) -> List[str]:
    filled_segments = list(segments)
    next_available: Optional[str] = None
    for index in range(len(filled_segments) - 1, -1, -1):
        current = filled_segments[index]
        if current is None:
            if next_available is None:
                raise ValueError("The last addressed segment is missing, so it cannot be filled from the next segment.")
            filled_segments[index] = next_available
        else:
            next_available = current
    return [segment for segment in filled_segments if segment is not None]


def decode_protected_dna_to_pixels(
    segments: Sequence[str],
    trees: Sequence[CodebookNode],
    check_size: int = 4,
    payload_length: int = 120,
    original_pixel_count: Optional[int] = None,
    codeword_length: int = 5,
    address_length: int = 8,
    expected_segment_count: Optional[int] = None,
    positions: Optional[Sequence[int]] = None,
) -> DecodedPixels:
    if expected_segment_count is None:
        expected_segment_count = len(segments)
    ordered_segments = fill_missing_segments_with_next(
        reorder_segments_by_address(
            segments=segments,
            expected_segment_count=expected_segment_count,
            address_length=address_length,
        )
    )
    decoded_segments = decode_segments_with_inner_rs(
        segments=ordered_segments,
        check_size=check_size,
        payload_length=payload_length,
        positions=positions,
    )
    pixels = decode_dna_to_pixels("".join(decoded_segments), trees, codeword_length=codeword_length)
    padded_pixel_count = len(pixels)
    if original_pixel_count is None:
        trimmed_pixels = pixels
    else:
        trimmed_pixels = pixels[:original_pixel_count]
    return DecodedPixels(
        pixels=trimmed_pixels,
        decoded_segments=decoded_segments,
        padded_pixel_count=padded_pixel_count,
        trimmed_pixel_count=len(trimmed_pixels),
        padding_pixels_removed=padded_pixel_count - len(trimmed_pixels),
        ordered_segments=ordered_segments,
    )
