
__all__ = [
    "CodebookNode",
    "DecodedPixels",
    "EncodedSegments",
    "TransformerDNA",
    "WorkflowConfig",
    "build_codebook_trees",
    "create_codebook",
    "decode_from_multiple_copies",
    "decode_dna_to_pixels",
    "decode_protected_dna_to_pixels",
    "encode_pixels_to_dna",
    "encode_segments_with_inner_rs",
    "reconstruct_sequence",
]


def __getattr__(name):
    if name in {"CodebookNode", "build_codebook_trees"}:
        from .codebook import CodebookNode, build_codebook_trees

        exports = {
            "CodebookNode": CodebookNode,
            "build_codebook_trees": build_codebook_trees,
        }
        return exports[name]

    if name in {"create_codebook", "encode_pixels_to_dna", "encode_segments_with_inner_rs", "EncodedSegments"}:
        from .encoder import EncodedSegments, create_codebook, encode_pixels_to_dna, encode_segments_with_inner_rs

        exports = {
            "EncodedSegments": EncodedSegments,
            "create_codebook": create_codebook,
            "encode_pixels_to_dna": encode_pixels_to_dna,
            "encode_segments_with_inner_rs": encode_segments_with_inner_rs,
        }
        return exports[name]

    if name in {"decode_dna_to_pixels", "decode_protected_dna_to_pixels", "DecodedPixels"}:
        from .decoder import DecodedPixels, decode_dna_to_pixels, decode_protected_dna_to_pixels

        exports = {
            "DecodedPixels": DecodedPixels,
            "decode_dna_to_pixels": decode_dna_to_pixels,
            "decode_protected_dna_to_pixels": decode_protected_dna_to_pixels,
        }
        return exports[name]

    if name in {"TransformerDNA", "reconstruct_sequence"}:
        from .transformer_model import TransformerDNA, reconstruct_sequence

        exports = {
            "TransformerDNA": TransformerDNA,
            "reconstruct_sequence": reconstruct_sequence,
        }
        return exports[name]

    if name in {"WorkflowConfig", "decode_from_multiple_copies"}:
        from .workflow import WorkflowConfig, decode_from_multiple_copies

        exports = {
            "WorkflowConfig": WorkflowConfig,
            "decode_from_multiple_copies": decode_from_multiple_copies,
        }
        return exports[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
