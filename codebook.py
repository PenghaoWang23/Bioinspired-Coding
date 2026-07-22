from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Sequence


DEFAULT_NUCLEOTIDES: Sequence[str] = ("A", "T", "G", "C")


@dataclass
class CodebookNode:
    value: str
    is_leaf: bool = False
    children: List[Optional["CodebookNode"]] = field(default_factory=lambda: [None] * 4)
    pixel_values: List[int] = field(default_factory=list)

    def iter_leaves(self) -> Iterator["CodebookNode"]:
        if self.is_leaf:
            yield self
            return
        for child in self.children:
            if child is not None:
                yield from child.iter_leaves()


def check_constraints(path: Sequence[str]) -> bool:
    counts = {base: 0 for base in DEFAULT_NUCLEOTIDES}
    for base in path:
        counts[base] += 1
    gc_count = counts["G"] + counts["C"]
    return not any(count > 4 for count in counts.values()) and 2 <= gc_count <= 3


def build_full_kary_tree(
    depth: int,
    nucleotides: Sequence[str],
    index: int = 0,
    path: Optional[Sequence[str]] = None,
) -> Optional[CodebookNode]:
    current_path = list(path or [])
    value = nucleotides[index % len(nucleotides)]

    if depth == 1:
        leaf = CodebookNode(value=value, is_leaf=True)
        return leaf if check_constraints(current_path + [value]) else None

    node = CodebookNode(value=value)
    current_path.append(value)
    for child_index in range(4):
        child = build_full_kary_tree(
            depth=depth - 1,
            nucleotides=nucleotides,
            index=index * 4 + child_index,
            path=current_path,
        )
        node.children[child_index] = child
    return node


def build_codebook_trees(
    depth: int = 5,
    nucleotides: Sequence[str] = DEFAULT_NUCLEOTIDES,
) -> List[CodebookNode]:
    trees: List[CodebookNode] = []
    for root_index in range(len(nucleotides)):
        tree = build_full_kary_tree(depth=depth, nucleotides=nucleotides, index=root_index)
        if tree is None:
            raise ValueError(f"Failed to build tree for root {nucleotides[root_index]}.")
        trees.append(tree)
    return trees


def assign_sequences_to_leaves(trees: Sequence[CodebookNode], pixel_values: Iterable[int]) -> None:
    leaves: List[CodebookNode] = []
    for tree in trees:
        leaves.extend(tree.iter_leaves())

    sorted_pixels = sorted({int(pixel) for pixel in pixel_values})
    if not sorted_pixels:
        return
    if len(sorted_pixels) > len(leaves):
        raise ValueError("Not enough valid leaf nodes to assign all pixel values.")

    leaf_index = 0
    for pixel_index, pixel in enumerate(sorted_pixels):
        remaining_leaves = len(leaves) - leaf_index
        remaining_pixels = len(sorted_pixels) - pixel_index
        group_size = max(1, remaining_leaves // remaining_pixels)
        for _ in range(group_size):
            if leaf_index >= len(leaves):
                break
            leaves[leaf_index].pixel_values = [pixel]
            leaf_index += 1


def assign_pixels_to_trees(
    trees: Sequence[CodebookNode],
    pixel_values_by_tree: Dict[int, Iterable[int]],
) -> None:
    for tree_index, pixel_values in pixel_values_by_tree.items():
        assign_sequences_to_leaves([trees[tree_index]], pixel_values)


def build_path(root: CodebookNode, target_value: int, path: Optional[List[str]] = None) -> Optional[List[str]]:
    prefix = list(path or [])
    if root.is_leaf and target_value in root.pixel_values:
        return prefix + [root.value]

    for child in root.children:
        if child is None:
            continue
        result = build_path(child, target_value, prefix + [root.value])
        if result is not None:
            return result
    return None


def build_pixel_lookup(trees: Sequence[CodebookNode]) -> Dict[int, str]:
    pixel_to_dna: Dict[int, str] = {}
    for tree in trees:
        for leaf in tree.iter_leaves():
            if not leaf.pixel_values:
                continue
            pixel = int(leaf.pixel_values[0])
            if pixel not in pixel_to_dna:
                path = build_path(tree, pixel)
                if path is not None:
                    pixel_to_dna[pixel] = "".join(path)
    return pixel_to_dna
