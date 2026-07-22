# Bioinspired-Coding Workflow
Encoding
Build a constrained DNA codebook, where each pixel value is mapped to a 5-nt DNA codeword.
Convert the image pixel stream into a DNA sequence using the codebook.
Pad the pixel stream so that the encoded DNA can be evenly divided into fixed-length payload segments.
Add inner Reed-Solomon (RS) parity to each payload segment for error correction.
Prepend a DNA address to each protected segment for later reordering.
Decoding
For each segment, collect multiple noisy sequencing copies.
Feed the multiple copies into the transformer model to reconstruct the addressed DNA segment.
Reorder reconstructed segments according to their DNA addresses.
If a segment is missing, fill it using the following segment.
Apply inner RS decoding to correct residual errors in each reordered segment.
Decode the corrected DNA sequence back into pixel values using the same 5-nt codebook.
Remove the padding added during encoding to recover the original pixel stream.
