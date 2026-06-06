"""
End-to-end tests covering the new robustness guarantees.

Run from project root with:
    python3 tests/test_e2e.py
"""

from __future__ import annotations

import os
import sys
# Make `src/` importable without needing PYTHONPATH or pip install.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import shutil
import threading
import numpy as np

from quorex.core.embeddings.encoder import Encoder
from quorex.core.vectordb.engine import VectorDBEngine


def fresh_dir(path: str) -> str:
    if os.path.exists(path):
        shutil.rmtree(path)
    return path


def assert_eq(label, actual, expected):
    ok = actual == expected
    mark = "✅" if ok else "❌"
    print(f"  {mark} {label}: actual={actual!r} expected={expected!r}")
    if not ok:
        raise AssertionError(label)


def assert_true(label, cond):
    mark = "✅" if cond else "❌"
    print(f"  {mark} {label}")
    if not cond:
        raise AssertionError(label)


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

events = [
    {"action": "viewed_pricing", "metadata": {"plan": "pro", "source": "dashboard"}},
    {"action": "upgraded_plan", "metadata": {"plan": "pro", "source": "billing"}},
    {"action": "clicked_cta", "metadata": {"source": "dashboard", "plan": "pro"}},
    {"action": "visited_homepage", "metadata": {"source": "organic"}},
    {"action": "searched_docs", "metadata": {"query": "api reference"}},
]
encoder = Encoder(n_components=5)
encoder.fit(events)
query_vec = encoder.encode({"action": "viewed_pricing", "metadata": {"plan": "pro"}})


# -----------------------------------------------------------------------
# Test 1 — WAL recovery WITHOUT a snapshot (proves vectors are in WAL).
# -----------------------------------------------------------------------

def test_wal_recovery_without_snapshot():
    print("\n=== Test 1 — WAL recovery without snapshot ===")
    path = fresh_dir("/tmp/quorex_test_t1")

    # Insert without ever triggering a checkpoint (checkpoint_every=999).
    engine = VectorDBEngine(path=path, dim=5, M=4, ef_construction=20,
                            ef_search=10, checkpoint_every=999)
    engine.start()
    for i, e in enumerate(events):
        engine.insert("u1", encoder.encode(e), e)
    # Simulate crash: close WAL without checkpoint.
    engine.wal.close()
    # No engine.stop() — no checkpoint written.

    assert_true("snapshot dir absent (never checkpointed)",
                not os.path.exists(os.path.join(path, "snapshot", "vectors.bin")))

    # Restart — recovery must come entirely from the WAL.
    engine2 = VectorDBEngine(path=path, dim=5, M=4, ef_construction=20, ef_search=10)
    engine2.start()
    assert_eq("vectors recovered from WAL", engine2.segment.total_vectors(), 5)
    hits = engine2.search("u1", query_vec, top_k=1)
    assert_true("query still works after pure WAL recovery", len(hits) == 1)
    assert_eq("top hit", hits[0]["meta"]["action"], "viewed_pricing")
    engine2.wal.close()


# -----------------------------------------------------------------------
# Test 2 — Hard delete persists across restart (no zombie zero-vectors).
# -----------------------------------------------------------------------

def test_hard_delete_persists():
    print("\n=== Test 2 — Hard delete persists ===")
    path = fresh_dir("/tmp/quorex_test_t2")

    with VectorDBEngine(path=path, dim=5, M=4, ef_construction=20,
                        ef_search=10, checkpoint_every=10) as engine:
        for i, e in enumerate(events):
            engine.insert("u1", encoder.encode(e), e)
        engine.delete("u1", 0)
        engine.delete("u1", 1)
        assert_eq("count after deletes", engine.segment.total_vectors(), 3)

    with VectorDBEngine(path=path, dim=5, M=4, ef_construction=20, ef_search=10) as engine2:
        assert_eq("count after restart", engine2.segment.total_vectors(), 3)
        live = engine2.segment.live_ids("u1")
        assert_true("vec_id 0 truly gone", 0 not in live)
        assert_true("vec_id 1 truly gone", 1 not in live)


# -----------------------------------------------------------------------
# Test 3 — Update API: persists + meta + vector both replace.
# -----------------------------------------------------------------------

def test_update_api():
    print("\n=== Test 3 — Update API ===")
    path = fresh_dir("/tmp/quorex_test_t3")

    new_meta = {"action": "viewed_pricing_starter", "metadata": {"plan": "starter"}}
    new_vec = encoder.encode(new_meta)

    with VectorDBEngine(path=path, dim=5, M=4, ef_construction=20,
                        ef_search=10, checkpoint_every=999) as engine:
        for e in events:
            engine.insert("u1", encoder.encode(e), e)
        ok = engine.update("u1", 0, new_vec, new_meta)
        assert_true("update returns True", ok)

    with VectorDBEngine(path=path, dim=5, M=4, ef_construction=20, ef_search=10) as engine2:
        meta = engine2.segment._metadata["u1"][0]
        assert_eq("updated meta persisted", meta["action"], "viewed_pricing_starter")


# -----------------------------------------------------------------------
# Test 4 — Compaction rebuilds HNSW cleanly after many deletes.
# -----------------------------------------------------------------------

def test_compaction():
    print("\n=== Test 4 — Compaction ===")
    path = fresh_dir("/tmp/quorex_test_t4")

    with VectorDBEngine(path=path, dim=5, M=4, ef_construction=20,
                        ef_search=10, checkpoint_every=999,
                        compact_after_deletes=2) as engine:
        for e in events:
            engine.insert("u1", encoder.encode(e), e)
        engine.delete("u1", 0)
        engine.delete("u1", 1)  # should auto-trigger compact (threshold=2)
        assert_eq("pending_deletes reset after auto-compact",
                  engine.segment.pending_deletes("u1"), 0)
        hits = engine.search("u1", query_vec, top_k=3)
        assert_true("search still works post-compact", len(hits) > 0)


# -----------------------------------------------------------------------
# Test 5 — Online encoder (partial_fit grows vocab without breaking).
# -----------------------------------------------------------------------

def test_online_encoder():
    print("\n=== Test 5 — Online encoder ===")
    enc = Encoder(n_components=5)
    enc.fit(events)
    initial_vocab = len(enc.vectorizer.vocabulary)

    new_events = [
        {"action": "clicked_checkout", "metadata": {"source": "mobile"}},
        {"action": "abandoned_cart", "metadata": {"source": "ios"}},
    ]
    added = enc.partial_fit(new_events)
    assert_true("new tokens added", added > 0)
    assert_eq("vocab grew", len(enc.vectorizer.vocabulary), initial_vocab + added)

    # Encoding old events still works
    old_vec = enc.encode(events[0])
    assert_eq("dim preserved", old_vec.shape, (5,))

    # Encoding new events doesn't crash (even if quality is low until refit)
    new_vec = enc.encode(new_events[0])
    assert_eq("new event encodes", new_vec.shape, (5,))

    # refit() rebuilds SVD on accumulated corpus
    enc.refit()
    new_vec_after = enc.encode(new_events[0])
    assert_true("post-refit vector is non-zero",
                float(np.linalg.norm(new_vec_after)) > 0)


# -----------------------------------------------------------------------
# Test 6 — Thread safety: concurrent inserts don't corrupt state.
# -----------------------------------------------------------------------

def test_thread_safety():
    print("\n=== Test 6 — Thread safety ===")
    path = fresh_dir("/tmp/quorex_test_t6")

    with VectorDBEngine(path=path, dim=5, M=4, ef_construction=20,
                        ef_search=10, checkpoint_every=50) as engine:

        def worker(uid, n):
            for i in range(n):
                ev = events[i % len(events)]
                engine.insert(uid, encoder.encode(ev), ev)

        threads = [threading.Thread(target=worker, args=(f"u{i}", 25)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        total = engine.segment.total_vectors()
        assert_eq("total inserts (4 threads * 25)", total, 100)
        for i in range(4):
            assert_eq(f"u{i} count", engine.segment.vector_count(f"u{i}"), 25)


# -----------------------------------------------------------------------
# Test 7 — Dirty flag prevents no-op checkpoints.
# -----------------------------------------------------------------------

def test_dirty_flag():
    print("\n=== Test 7 — Dirty flag ===")
    path = fresh_dir("/tmp/quorex_test_t7")

    with VectorDBEngine(path=path, dim=5) as engine:
        for e in events:
            engine.insert("u1", encoder.encode(e), e)
        engine.checkpoint()
        size_before = sum(engine.storage.size_bytes().values())
        engine.checkpoint()  # should be a no-op
        size_after = sum(engine.storage.size_bytes().values())
        assert_eq("no-op checkpoint changes nothing", size_before, size_after)


# -----------------------------------------------------------------------
# Run all
# -----------------------------------------------------------------------

if __name__ == "__main__":
    test_wal_recovery_without_snapshot()
    test_hard_delete_persists()
    test_update_api()
    test_compaction()
    test_online_encoder()
    test_thread_safety()
    test_dirty_flag()
    print("\nAll tests passed.")
