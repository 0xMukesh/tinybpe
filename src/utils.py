import os
from typing import BinaryIO


def find_chunk_boundaries(
    file: BinaryIO, num_chunks: int, split_token: bytes
) -> list[int]:
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // num_chunks

    chunk_boundaries = [i * chunk_size for i in range(num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_read_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)

        while True:
            mini_chunk = file.read(mini_chunk_read_size)

            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break

            initial_position += mini_chunk_read_size

    return sorted(set(chunk_boundaries))
