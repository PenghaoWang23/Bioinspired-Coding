# Bioinspired-Coding Workflow
Encoding
1.Build a constrained DNA codebook, where each pixel value is mapped to a 5-nt DNA codeword.
2.Convert the image pixel stream into a DNA sequence using the codebook.
3.Pad the pixel stream so that the encoded DNA can be evenly divided into fixed-length payload segments.
4.Add inner Reed-Solomon (RS) parity to each payload segment for error correction.
5.Prepend a DNA address to each protected segment for later reordering.
Decoding
1.For each segment, collect multiple noisy sequencing copies.
2.Feed the multiple copies into the transformer model to reconstruct the addressed DNA segment.
3.Reorder reconstructed segments according to their DNA addresses.
4.If a segment is missing, fill it using the following segment.
5.Apply inner RS decoding to correct residual errors in each reordered segment.
6.Decode the corrected DNA sequence back into pixel values using the same 5-nt codebook.
7.Remove the padding added during encoding to recover the original pixel stream.
