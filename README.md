# Core DNA Workflow

This directory now contains a cleaned core workflow for the parts you want to publish:

- `codebook.py`: codebook tree construction and pixel-to-codeword assignment.
- `encoder.py`: pixel-to-DNA encoding, explicit segment padding, inner RS protection, and address prefixing.
- `transformer_model.py`: transformer definition and copy-consensus reconstruction.
- `decoder.py`: transformer-first decoding, address-based reordering, missing-segment filling, and DNA-to-pixel restoration.
- `workflow.py`: minimal high-level API that connects the real encode/decode flow.

The following files are legacy or data-processing oriented and are intentionally not part of the cleaned GitHub-facing workflow:

- `kmeans_image_segmentation.py`
- `RGB.py`
- `introduce_errors.py`
- `main.py`
- `rs_codec&data_datahandle.py`
- `encode_image_dna.py`
- `decode_dna_image.py`

Core dependencies:

- `reedsolo` for RS protection and correction.
- `torch` for the transformer reconstruction module only.

Install them with:

```bash
pip install -r main_dont_move_github/requirements-core.txt
```

Workflow summary:

1. Build four constrained codebook trees rooted at `A/T/G/C`.
2. Assign pixel values to leaf nodes.
3. Encode pixels into 5-nt codewords.
4. Pad the pixel stream so the encoded DNA can be split into fixed 120-nt payload segments.
5. Add inner RS parity to each payload segment, then prepend a fixed DNA address to every segment.
6. Feed multiple noisy copies of each addressed segment into the transformer to reconstruct that segment.
7. Reorder the reconstructed segments by address and fill any missing segment with the next segment.
8. Run RS-constrained recovery and map 5-nt codewords back to pixels.
9. Remove the padding pixels that were added during encoding.

Minimal usage:

```python
from main_dont_move_github.workflow import (
    WorkflowConfig,
    build_codebook_from_pixels,
    decode_from_multiple_copies,
    encode_pixels,
)

config = WorkflowConfig()
trees = build_codebook_from_pixels({
    0: [0, 1, 2],
    1: [3, 4, 5],
    2: [6, 7, 8],
    3: [9, 10, 11],
})

encoded = encode_pixels([0, 3, 6, 9], trees, config=config)

# Each addressed segment has multiple noisy copies.
segment_copies = [
    [segment, segment, segment]
    for segment in encoded.protected_segments
]

decoded = decode_from_multiple_copies(
    segment_copies=segment_copies,
    trees=trees,
    config=config,
    original_pixel_count=encoded.original_pixel_count,
    model=model,
    device=device,
)

decoded_pixels = decoded.pixels
```
