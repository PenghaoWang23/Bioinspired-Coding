from __future__ import annotations

import itertools
import math
from collections import Counter
from typing import Iterable, List, Optional, Sequence

import reedsolo


DNA_TO_BITS = {"A": "00", "T": "01", "C": "10", "G": "11"}
BITS_TO_DNA = {value: key for key, value in DNA_TO_BITS.items()}


def dna_to_binary(dna_sequence: str) -> str:
    return "".join(DNA_TO_BITS[base] for base in dna_sequence)


def binary_to_dna(bit_string: str) -> str:
    if len(bit_string) % 2 != 0:
        raise ValueError("Bit string length must be even.")
    return "".join(BITS_TO_DNA[bit_string[index:index + 2]] for index in range(0, len(bit_string), 2))


def binary_to_byte_list(bit_string: str) -> List[int]:
    if len(bit_string) % 8 != 0:
        raise ValueError("Bit string length must be divisible by 8.")
    return [int(bit_string[index:index + 8], 2) for index in range(0, len(bit_string), 8)]


def byte_list_to_binary(byte_values: Iterable[int]) -> str:
    return "".join(format(one_byte, "08b") for one_byte in byte_values)


def int_to_dna_address(value: int, address_length: int = 8) -> str:
    if value < 0:
        raise ValueError("Address value must be non-negative.")
    max_value = 4 ** address_length
    if value >= max_value:
        raise ValueError(f"Address value {value} exceeds capacity of {address_length} bases.")
    return binary_to_dna(format(value, f"0{address_length * 2}b"))


def dna_address_to_int(address_dna: str) -> int:
    return int(dna_to_binary(address_dna), 2)


def expected_segment_count(
    pixel_count: int,
    payload_length: int,
    codeword_length: int = 5,
) -> int:
    pixels_per_segment = payload_length // codeword_length
    if pixels_per_segment <= 0:
        raise ValueError("payload_length must be at least one codeword.")
    return math.ceil(pixel_count / pixels_per_segment) if pixel_count > 0 else 0


def build_inner_rs_positions(payload_length: int, stride: int = 10) -> List[int]:
    positions: List[int] = []
    for start in range(0, payload_length, stride):
        positions.extend([start + 1, start + 2, start + 6, start + 7])
    return [position for position in positions if position <= payload_length]


def encode_inner_rs(dna_sequence: str, positions: Sequence[int], check_size: int) -> str:
    extracted_sequence = "".join(dna_sequence[position - 1] for position in positions)
    encoded_bytes = list(reedsolo.RSCodec(check_size).encode(binary_to_byte_list(dna_to_binary(extracted_sequence))))
    rs_bits = byte_list_to_binary(encoded_bytes)[-check_size * 8:]
    return dna_sequence + binary_to_dna(rs_bits)


def decode_inner_rs(
    mutated_dna: str,
    positions: Sequence[int],
    check_size: int,
    payload_length: Optional[int] = None,
) -> Optional[str]:
    payload_end = payload_length or (len(mutated_dna) - check_size * 4)
    extracted_sequence = "".join(mutated_dna[position - 1] for position in positions)
    rs_code_dna = mutated_dna[-check_size * 4:]
    input_bytes = binary_to_byte_list(dna_to_binary(extracted_sequence + rs_code_dna))

    try:
        decoded = list(reedsolo.RSCodec(check_size).decode(input_bytes))
    except reedsolo.ReedSolomonError:
        return None

    decoded_payload = decoded[0] if decoded and not isinstance(decoded[0], int) else decoded
    corrected_bits = byte_list_to_binary(decoded_payload)
    corrected_dna = binary_to_dna(corrected_bits)

    corrected_full_dna = list(mutated_dna)
    for index, position in enumerate(positions):
        corrected_full_dna[position - 1] = corrected_dna[index]
    return "".join(corrected_full_dna[:payload_end])


def viterbi_decode_with_rs(
    sequences: Sequence[str],
    positions: Sequence[int],
    mutated_dna: str,
    check_size: int,
    payload_length: Optional[int] = None,
    threshold: float = 0.1,
) -> str:
    if len(positions) % 2 != 0:
        raise ValueError("The number of positions must be even.")

    pair_sets: List[List[str]] = []
    for index in range(0, len(positions), 2):
        observed_pairs = []
        for sequence in sequences:
            left = positions[index] - 1
            right = positions[index + 1] - 1
            if right < len(sequence):
                observed_pairs.append(sequence[left] + sequence[right])
        pair_counter = Counter(observed_pairs)
        candidate_pairs = [
            pair for pair, count in pair_counter.items()
            if observed_pairs and count / len(observed_pairs) >= threshold
        ]
        pair_sets.append(sorted(candidate_pairs, key=lambda pair: pair_counter[pair], reverse=True))

    rs_code_candidates = sorted({sequence[-check_size * 4:] for sequence in sequences if len(sequence) >= check_size * 4})
    pair_sets.append(rs_code_candidates)

    payload_end = payload_length or (len(mutated_dna) - check_size * 4)
    rs_codec = reedsolo.RSCodec(check_size)

    for combination in itertools.product(*pair_sets):
        candidate_dna = "".join(combination)
        try:
            decoded = list(rs_codec.decode(binary_to_byte_list(dna_to_binary(candidate_dna))))
        except reedsolo.ReedSolomonError:
            continue

        decoded_payload = decoded[0] if decoded and not isinstance(decoded[0], int) else decoded
        corrected_bits = byte_list_to_binary(decoded_payload)
        corrected_dna = binary_to_dna(corrected_bits)

        corrected_full_dna = list(mutated_dna)
        for index, position in enumerate(positions):
            corrected_full_dna[position - 1] = corrected_dna[index]
        return "".join(corrected_full_dna[:payload_end])

    return mutated_dna[:payload_end]
