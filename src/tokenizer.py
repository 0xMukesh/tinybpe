import heapq
import itertools
import pickle
from collections import Counter, defaultdict
from enum import Enum
from multiprocessing import Pool

import regex as re
from tqdm import tqdm
from typing_extensions import BinaryIO

from src.utils import find_chunk_boundaries

DEFAULT_TOKENIZER_STATE_PATH = "tokenizer.pkl"
GPT2_SPLIT_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


def _iter_chunks(f: BinaryIO, chunk_boundaries: list[int]):
    for start, end in itertools.pairwise(chunk_boundaries):
        f.seek(start)
        yield f.read(end - start).decode("utf-8")


class SplitPattern(Enum):
    GPT2 = GPT2_SPLIT_PATTERN
    GPT4 = GPT4_SPLIT_PATTERN

    @property
    def regex(self) -> str:
        return self.value


class Tokenizer:
    def __init__(
        self,
        split_pattern: SplitPattern,
        special_tokens: dict[str, int],
    ) -> None:
        self.merges: dict[tuple[int, int], int] = defaultdict()
        self.vocab: dict[int, bytes] = {idx: bytes([idx]) for idx in range(256)}
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}
        self.split_pattern = re.compile(split_pattern.regex)
        self.special_tokens_pattern = (
            "(" + "|".join(re.escape(k) for k in special_tokens) + ")"
        )

    def _pretokenize(self, text: str):
        pair_counts: Counter[tuple[int, int]] = Counter()
        sequences: list[list[int]] = []

        for part in re.split(self.special_tokens_pattern, text):
            if part in self.special_tokens:
                continue

            for m in re.findall(self.split_pattern, part):
                ids: list[int] = m.encode("utf-8")
                sequences.append(ids)

                for pair in itertools.pairwise(ids):
                    pair_counts[pair] += 1

        return list(pair_counts.items()), sequences

    def train(
        self,
        input_path: str,
        vocab_size: int,
        split_token: bytes,
        n_workers: int,
        n_chunks_per_worker: int,
        use_imap_iter: bool = False,
    ):
        assert vocab_size >= 256

        with open(input_path, "rb") as f:
            chunk_boundaries = find_chunk_boundaries(
                f, n_workers * n_chunks_per_worker, split_token
            )

            if not use_imap_iter:
                chunks = []
                for start, end in itertools.pairwise(chunk_boundaries):
                    f.seek(start)
                    chunks.append(f.read(end - start).decode("utf-8"))
            else:
                chunks = _iter_chunks(f, chunk_boundaries)

            pair_to_count: Counter[tuple[int, int]] = Counter()
            seq_to_count: Counter[tuple[int, ...]] = Counter()

            # parallel pretokenization
            with Pool(
                processes=n_workers,
            ) as p:
                for results in p.imap(self._pretokenize, chunks):
                    for pair, count in results[0]:
                        pair_to_count[pair] += count

                    for seq in results[1]:
                        seq_to_count[tuple(seq)] += 1

        all_seqs = [list(seq) for seq in seq_to_count]  # deduped seqs
        token_to_seq_ids: dict[int, set[int]] = defaultdict(set[int])  # inverse index

        for seq_id, seq in enumerate(all_seqs):
            for tok in seq:
                token_to_seq_ids[tok].add(seq_id)

        heap = [
            (-count, pair) for pair, count in pair_to_count.items()
        ]  # priority queue
        heapq.heapify(heap)

        for i in tqdm(range(vocab_size - 256)):
            if not pair_to_count:
                break

            while heap:
                neg_count, best_pair = heapq.heappop(heap)
                if (
                    pair_to_count.get(best_pair, 0) == -neg_count
                ):  # to avoid stale entries
                    break
            else:
                break

            new_token_idx = 256 + i
            a, b = best_pair

            self.merges[best_pair] = new_token_idx
            self.vocab[new_token_idx] = self.vocab[a] + self.vocab[b]
            del pair_to_count[best_pair]

            affected_seqs = token_to_seq_ids[a] & token_to_seq_ids[b]

            for seq_id in affected_seqs:
                seq = all_seqs[seq_id]
                freq = seq_to_count[tuple(seq)]
                new_seq: list[int] = []
                j = 0
                n_seq = len(seq)

                while j < n_seq:
                    if j < n_seq - 1 and seq[j] == a and seq[j + 1] == b:
                        if j > 0:
                            pair_to_count[(seq[j - 1], a)] -= freq
                            new_pair = (seq[j - 1], new_token_idx)
                            pair_to_count[new_pair] += freq

                            # push the latest counts to the priority queue
                            heapq.heappush(heap, (-pair_to_count[new_pair], new_pair))
                        if j < n_seq - 2:
                            pair_to_count[(b, seq[j + 2])] -= freq
                            new_pair = (new_token_idx, seq[j + 2])
                            pair_to_count[new_pair] += freq

                            heapq.heappush(heap, (-pair_to_count[new_pair], new_pair))

                        new_seq.append(new_token_idx)
                        j += 2
                    else:
                        new_seq.append(seq[j])
                        j += 1

                all_seqs[seq_id] = new_seq

                if a not in new_seq:
                    token_to_seq_ids[a].discard(seq_id)
                if b not in new_seq:
                    token_to_seq_ids[b].discard(seq_id)
                token_to_seq_ids[new_token_idx].add(seq_id)

    def _encode_bytes(self, text_bytes: bytes) -> list[int]:
        tokens = list(text_bytes)
        pair_to_count = Counter(itertools.pairwise(tokens))
        heap: list[tuple[int | float, tuple[int, int]]] = [
            (
                self.merges.get(pair, float("inf")),
                pair,
            )  # start encoding by merge tokens with least token idx
            for pair, _ in pair_to_count.items()
        ]
        heapq.heapify(heap)

        while True:
            if len(pair_to_count) == 0:
                break

            token_idx, to_merge = heapq.heappop(heap)
            if token_idx == float("inf"):
                break

            token_idx = int(token_idx)

            i, j = 0, 0
            n_tokens = len(tokens)
            new_tokens: list[int] = [0] * n_tokens

            while i < n_tokens:
                if (
                    i < n_tokens - 1
                    and tokens[i] == to_merge[0]
                    and tokens[i + 1] == to_merge[1]
                ):
                    if i > 0:
                        pair_to_count[(tokens[i - 1], tokens[i])] -= 1
                        pair_to_count[(tokens[i - 1], token_idx)] += 1
                    if i < n_tokens - 2:
                        pair_to_count[(tokens[i], tokens[i + 2])] -= 1
                        pair_to_count[(token_idx, tokens[i])] += 1

                    new_tokens[j] = token_idx
                    j += 1
                    i += 2
                else:
                    new_tokens[j] = tokens[i]
                    j += 1
                    i += 1

        return tokens

    def encode(self, text: str) -> list[int]:
        tokens: list[int] = []

        for part in re.split(self.special_tokens_pattern, text):
            if not part:
                continue

            if part in self.special_tokens:
                tokens.append(self.special_tokens[part])
            else:
                for m in re.findall(self.split_pattern, part):
                    tokens.extend(self._encode_bytes(m.encode("utf-8")))

        return tokens

    def decode(self, tokens: list[int]) -> str:
        part_bytes: list[bytes] = []

        for idx in tokens:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))

        return b"".join(part_bytes).decode("utf-8")

    def _save_state(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"merges": self.merges, "vocab": self.vocab}, f)

    def _load_state(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.merges = data["merges"]
            self.vocab = data["vocab"]
