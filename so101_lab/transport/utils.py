# Adapted from: lerobot (https://github.com/huggingface/lerobot)
# Original license: Apache-2.0
# Changes: Standalone — no lerobot imports; simplified receive_bytes_in_chunks
#          (no queue/shutdown_event); works in both Python 3.11 and 3.12.

"""Chunked gRPC transport utilities for PPO training."""

import io
import json
import logging
import pickle  # nosec B403: internal serialization only
from typing import Any

CHUNK_SIZE = 2 * 1024 * 1024  # 2 MB
MAX_MESSAGE_SIZE = 4 * 1024 * 1024  # 4 MB


def send_bytes_in_chunks(buffer: bytes, message_class: Any):
    """Yield chunked gRPC messages from a bytes buffer."""
    bytes_io = io.BytesIO(buffer)
    bytes_io.seek(0, io.SEEK_END)
    total = bytes_io.tell()
    bytes_io.seek(0)

    sent = 0
    while sent < total:
        if sent == 0:
            state = 1  # TRANSFER_BEGIN
        elif sent + CHUNK_SIZE >= total:
            state = 3  # TRANSFER_END
        else:
            state = 2  # TRANSFER_MIDDLE

        chunk = bytes_io.read(min(CHUNK_SIZE, total - sent))
        yield message_class(transfer_state=state, data=chunk)
        sent += len(chunk)


def receive_bytes_in_chunks(iterator) -> bytes:
    """Reassemble bytes from a streaming iterator of DataChunk messages."""
    buf = io.BytesIO()
    for item in iterator:
        if item.transfer_state == 1:  # TRANSFER_BEGIN
            buf.seek(0)
            buf.truncate(0)
            buf.write(item.data)
        elif item.transfer_state == 2:  # TRANSFER_MIDDLE
            buf.write(item.data)
        elif item.transfer_state == 3:  # TRANSFER_END
            buf.write(item.data)
            return buf.getvalue()
        else:
            raise ValueError(f"Unknown transfer state: {item.transfer_state}")
    return buf.getvalue()


def grpc_channel_options(
    max_receive_message_length: int = MAX_MESSAGE_SIZE,
    max_send_message_length: int = MAX_MESSAGE_SIZE,
) -> list:
    """gRPC channel options with retry config."""
    service_config = {
        "methodConfig": [
            {
                "name": [{}],
                "retryPolicy": {
                    "maxAttempts": 5,
                    "initialBackoff": "0.1s",
                    "maxBackoff": "2s",
                    "backoffMultiplier": 2,
                    "retryableStatusCodes": ["UNAVAILABLE", "DEADLINE_EXCEEDED"],
                },
            }
        ]
    }
    return [
        ("grpc.max_receive_message_length", max_receive_message_length),
        ("grpc.max_send_message_length", max_send_message_length),
        ("grpc.enable_retries", 1),
        ("grpc.service_config", json.dumps(service_config)),
    ]


def serialize(obj: Any) -> bytes:
    """Pickle-serialize a Python object."""
    return pickle.dumps(obj)


def deserialize(data: bytes) -> Any:
    """Pickle-deserialize bytes."""
    return pickle.loads(data)  # nosec B301: internal use only
