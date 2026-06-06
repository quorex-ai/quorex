from __future__ import annotations

import json
import os
import struct
import time
import numpy as np
from enum import Enum
from dataclasses import dataclass, asdict, field


class OpType(str, Enum):
    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    CHECKPOINT = "CHECKPOINT"  # marks a full snapshot point


@dataclass
class WALEntry:
    op: str                              # OpType value
    user_id: str
    vec_id: int
    meta: dict
    timestamp: float                     # unix timestamp
    vector: list | None = field(default=None)  # float list — None for DELETE / CHECKPOINT


class WAL:
    def __init__(self, path: str):
        """
        path: path to the WAL file (e.g. "./data/quorex.wal")

        Binary layout per entry:
            [4 bytes: payload length][N bytes: JSON payload]
        Payload is the WALEntry dataclass serialized as JSON.

        For INSERT/UPDATE entries, the vector is encoded as a list of
        Python floats (JSON-native). This makes the WAL self-sufficient
        for recovery — engine.replay() can fully reconstruct state from
        the WAL alone, with or without a snapshot.
        """
        self.path = path
        self._file = None
        self._entry_count = 0

        # Cached offset of the last CHECKPOINT entry in the WAL file.
        # Updated on every log_checkpoint() and on the first replay() so
        # truncate_after_checkpoint() can seek directly rather than
        # re-reading the entire log.
        self._last_checkpoint_offset: int | None = None
        self._last_checkpoint_index: int = -1

    # Lifecycle

    def open(self) -> None:
        """Opens the WAL file in append+binary mode."""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._file = open(self.path, "ab")

    def close(self) -> None:
        """Flushes and closes the WAL file."""
        if self._file:
            self._file.flush()
            os.fsync(self._file.fileno())
            self._file.close()
            self._file = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    # Write

    def log_insert(
        self,
        user_id: str,
        vec_id: int,
        meta: dict,
        vector: np.ndarray,
    ) -> None:
        """Logs an INSERT with its vector — full recovery without snapshot."""
        entry = WALEntry(
            op=OpType.INSERT,
            user_id=user_id,
            vec_id=vec_id,
            meta=meta,
            timestamp=time.time(),
            vector=vector.astype(np.float32).tolist(),
        )
        self._write(entry)

    def log_update(
        self,
        user_id: str,
        vec_id: int,
        meta: dict,
        vector: np.ndarray,
    ) -> None:
        """Logs an UPDATE — same payload shape as INSERT."""
        entry = WALEntry(
            op=OpType.UPDATE,
            user_id=user_id,
            vec_id=vec_id,
            meta=meta,
            timestamp=time.time(),
            vector=vector.astype(np.float32).tolist(),
        )
        self._write(entry)

    def log_delete(self, user_id: str, vec_id: int) -> None:
        """Logs a DELETE operation."""
        entry = WALEntry(
            op=OpType.DELETE,
            user_id=user_id,
            vec_id=vec_id,
            meta={},
            timestamp=time.time(),
            vector=None,
        )
        self._write(entry)

    def log_checkpoint(self) -> None:
        """Logs a CHECKPOINT — marks a safe recovery point."""
        entry = WALEntry(
            op=OpType.CHECKPOINT,
            user_id="__system__",
            vec_id=-1,
            meta={},
            timestamp=time.time(),
            vector=None,
        )
        offset_before = self._file.tell() if self._file else None
        self._write(entry)
        if offset_before is not None:
            self._last_checkpoint_offset = offset_before
            self._last_checkpoint_index = self._entry_count - 1

    def _write(self, entry: WALEntry) -> None:
        """fsync after every write — guarantees durability on crash."""
        if not self._file:
            raise RuntimeError("WAL is not open. Call open() first.")

        payload = json.dumps(asdict(entry)).encode("utf-8")
        length = struct.pack(">I", len(payload))

        self._file.write(length)
        self._file.write(payload)
        self._file.flush()
        os.fsync(self._file.fileno())

        self._entry_count += 1

    # Recovery

    def replay(self) -> list[WALEntry]:
        """
        Reads all valid entries from the WAL file.
        Stops at the last valid entry — ignores corrupted tail.

        Also updates the cached last_checkpoint position so subsequent
        calls don't need to re-scan.
        """
        entries = []
        last_cp_offset = None
        last_cp_index = -1

        if not os.path.exists(self.path):
            self._last_checkpoint_offset = None
            self._last_checkpoint_index = -1
            return entries

        with open(self.path, "rb") as f:
            while True:
                entry_offset = f.tell()
                length_bytes = f.read(4)
                if len(length_bytes) < 4:
                    break

                length = struct.unpack(">I", length_bytes)[0]

                payload_bytes = f.read(length)
                if len(payload_bytes) < length:
                    break

                try:
                    data = json.loads(payload_bytes.decode("utf-8"))
                    entry = WALEntry(**data)
                    entries.append(entry)
                    if entry.op == OpType.CHECKPOINT:
                        last_cp_offset = entry_offset
                        last_cp_index = len(entries) - 1
                except (json.JSONDecodeError, TypeError):
                    break  # corrupted entry — stop replay

        self._last_checkpoint_offset = last_cp_offset
        self._last_checkpoint_index = last_cp_index
        return entries

    def last_checkpoint(self) -> int:
        """
        Returns the index of the last CHECKPOINT entry, or -1 if none.
        Uses cache when available; falls back to replay() on cold start.
        """
        if self._last_checkpoint_index >= 0 or self._last_checkpoint_offset is None:
            # If cache is populated, trust it. Otherwise fall through to scan.
            if self._last_checkpoint_index >= 0:
                return self._last_checkpoint_index
        self.replay()
        return self._last_checkpoint_index

    def truncate_after_checkpoint(self) -> int:
        """
        Removes everything up to and including the last CHECKPOINT.
        Uses the cached byte offset to avoid a full replay+rewrite when
        possible.

        Returns the number of entries kept after truncation.
        """
        if not os.path.exists(self.path):
            return 0

        # Make sure cache is populated.
        if self._last_checkpoint_offset is None:
            self.replay()

        cp_offset = self._last_checkpoint_offset
        if cp_offset is None:
            return self._entry_count  # nothing to truncate

        # Compute the byte position immediately AFTER the checkpoint entry.
        with open(self.path, "rb") as f:
            f.seek(cp_offset)
            length_bytes = f.read(4)
            if len(length_bytes) < 4:
                return self._entry_count
            length = struct.unpack(">I", length_bytes)[0]
            keep_start = cp_offset + 4 + length
            f.seek(keep_start)
            tail = f.read()

        # Close and rewrite the file with only the tail.
        was_open = self._file is not None
        if was_open:
            self.close()

        with open(self.path, "wb") as f:
            f.write(tail)
            f.flush()
            os.fsync(f.fileno())

        # Reset caches — the checkpoint was removed.
        self._last_checkpoint_offset = None
        self._last_checkpoint_index = -1

        # Re-count remaining entries cheaply.
        kept = 0
        with open(self.path, "rb") as f:
            while True:
                lb = f.read(4)
                if len(lb) < 4:
                    break
                ln = struct.unpack(">I", lb)[0]
                if len(f.read(ln)) < ln:
                    break
                kept += 1
        self._entry_count = kept

        if was_open:
            self.open()

        return kept

    # Utils

    def size_bytes(self) -> int:
        if not os.path.exists(self.path):
            return 0
        return os.path.getsize(self.path)

    def __repr__(self) -> str:
        return f"WAL(path={self.path}, size={self.size_bytes()}B)"


if __name__ == "__main__":
    wal_path = "/tmp/quorex_test.wal"

    if os.path.exists(wal_path):
        os.remove(wal_path)

    print("--- Writing to WAL ---")
    with WAL(wal_path) as wal:
        wal.log_insert("user_123", 0, {"action": "viewed_pricing"}, np.array([0.1, 0.2, 0.3], dtype=np.float32))
        wal.log_insert("user_123", 1, {"action": "upgraded_plan"}, np.array([0.4, 0.5, 0.6], dtype=np.float32))
        wal.log_insert("user_456", 0, {"action": "visited_homepage"}, np.array([0.7, 0.8, 0.9], dtype=np.float32))
        wal.log_checkpoint()
        wal.log_insert("user_123", 2, {"action": "clicked_cta"}, np.array([0.2, 0.3, 0.4], dtype=np.float32))
        wal.log_delete("user_123", 0)
        print(wal)

    print("\n--- Replaying WAL ---")
    wal = WAL(wal_path)
    entries = wal.replay()
    for e in entries:
        vec_preview = e.vector[:3] if e.vector else None
        print(f"  [{e.op}] user={e.user_id} vec_id={e.vec_id} vector={vec_preview} meta={e.meta}")

    print(f"\nLast checkpoint at index: {wal.last_checkpoint()}")

    print("\n--- Compacting WAL (drops everything through last checkpoint) ---")
    kept = wal.truncate_after_checkpoint()
    print(f"Entries kept after compaction: {kept}")

    entries = wal.replay()
    for e in entries:
        vec_preview = e.vector[:3] if e.vector else None
        print(f"  [{e.op}] user={e.user_id} vec_id={e.vec_id} vector={vec_preview} meta={e.meta}")
