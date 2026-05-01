import heapq
import itertools
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

type _Token = int
type _Pair = tuple[_Token, _Token]
type _Seq = list[_Token]

_worker_pattern = None
_worker_special_tokens_pattern = None
_worker_special_tokens = None


def _init_worker(split_pattern: str, special_tokens_pattern: str, special_tokens: set):
    global _worker_pattern, _worker_special_tokens_pattern, _worker_special_tokens
    _worker_pattern = re.compile(split_pattern)
    _worker_special_tokens_pattern = special_tokens_pattern
    _worker_special_tokens = special_tokens


def _iter_chunks(f: BinaryIO, chunk_boundaries: list[int]):
    for start, end in itertools.pairwise(chunk_boundaries):
        f.seek(start)
        yield f.read(end - start).decode()


def _pretokenize(text: str):
    pair_counts: Counter[_Pair] = Counter()
    sequences: list[_Seq] = []

    if (
        _worker_pattern is None
        or _worker_special_tokens_pattern is None
        or _worker_special_tokens is None
    ):
        return []

    for part in re.split(_worker_special_tokens_pattern, text):
        if part in _worker_special_tokens:
            continue
        for m in re.findall(_worker_pattern, part):
            ids = m.encode("utf-8")

            sequences.append(ids)
            for pair in zip(ids, ids[1:]):
                pair_counts[pair] += 1

    return list(pair_counts.items()), sequences


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
        special_tokens: dict[str, _Token],
    ) -> None:
        self.merges: dict[_Pair, int] = defaultdict()
        self.vocab: dict[_Token, bytes] = {idx: bytes([idx]) for idx in range(256)}
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}
        self.split_pattern = re.compile(split_pattern.regex)
        self.special_tokens_pattern = (
            "(" + "|".join(re.escape(k) for k in special_tokens) + ")"
        )

    def train(
        self,
        input_path: str,
        vocab_size: int,
        split_token: bytes,
        n_workers: int,
        use_imap_iter: bool = False,
    ):
        assert vocab_size >= 256

        with open(input_path, "rb") as f:
            chunk_boundaries = find_chunk_boundaries(f, n_workers, split_token)

            if not use_imap_iter:
                chunks = []

                for start, end in itertools.pairwise(chunk_boundaries):
                    f.seek(start)
                    chunks.append(f.read(end - start).decode("utf-8"))
            else:
                chunks = _iter_chunks(f, chunk_boundaries)

            init_args = (
                self.split_pattern,
                self.special_tokens_pattern,
                set(self.special_tokens),
            )

            pair_to_count: Counter[_Pair] = Counter()
            seq_to_count: Counter[tuple[_Token, ...]] = Counter()  # for dedup

            # parallel pretokenization
            with Pool(
                processes=n_workers, initializer=_init_worker, initargs=init_args
            ) as p:
                for results in p.imap(_pretokenize, chunks, chunksize=2):
                    for pair, count in results[0]:
                        pair_to_count[pair] += count

                    for seq in results[1]:
                        seq_to_count[tuple(seq)] += 1

        all_seqs = [list(seq) for seq in seq_to_count]
        token_to_seq_ids: dict[_Token, set[int]] = defaultdict(set[int])

        for seq_id, seq in enumerate(all_seqs):
            for tok in seq:
                token_to_seq_ids[tok].add(seq_id)

        heap = [(-count, pair) for pair, count in pair_to_count.items()]
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

                while j < len(seq):
                    if j < len(seq) - 1 and seq[j] == a and seq[j + 1] == b:
                        if j > 0:
                            pair_to_count[(seq[j - 1], a)] -= freq
                            new_pair = (seq[j - 1], new_token_idx)
                            pair_to_count[new_pair] += freq

                            # push the latest counts to the priority queue
                            heapq.heappush(heap, (-pair_to_count[new_pair], new_pair))
                        if j < len(seq) - 2:
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
