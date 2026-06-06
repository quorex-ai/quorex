"""
Interactive console for the Quorex vector DB.

Usage:
    python3 tests/console.py

You'll get a prompt where you can type commands to exercise the engine
live. The encoder is fitted on a seed corpus and is extended online as
you insert new events. Type `help` to see available commands.
"""

from __future__ import annotations

import os
import sys
import shutil

# Make `src/` importable without PYTHONPATH or pip install.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quorex.core.embeddings.encoder import Encoder
from quorex.core.vectordb.engine import VectorDBEngine


# -----------------------------------------------------------------------
# Visual helpers
# -----------------------------------------------------------------------

class C:
    """ANSI color codes — kept minimal."""
    RESET = "\033[0m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    MAGENTA = "\033[35m"


def banner(text: str) -> None:
    print(f"\n{C.CYAN}━━━ {text} ━━━{C.RESET}")


def step(text: str) -> None:
    print(f"  {C.DIM}→{C.RESET} {text}")


def ok(text: str) -> None:
    print(f"  {C.GREEN}✓{C.RESET} {text}")


def warn(text: str) -> None:
    print(f"  {C.YELLOW}!{C.RESET} {text}")


def err(text: str) -> None:
    print(f"  {C.RED}✗{C.RESET} {text}")


# -----------------------------------------------------------------------
# Seed data — used to fit the encoder on first start
# -----------------------------------------------------------------------

SEED_EVENTS = [
    {"action": "viewed_pricing", "metadata": {"plan": "pro", "source": "dashboard"}},
    {"action": "upgraded_plan", "metadata": {"plan": "pro", "source": "billing"}},
    {"action": "clicked_cta", "metadata": {"source": "dashboard", "plan": "pro"}},
    {"action": "visited_homepage", "metadata": {"source": "organic"}},
    {"action": "searched_docs", "metadata": {"query": "api reference"}},
    {"action": "opened_onboarding", "metadata": {"source": "welcome_email"}},
    {"action": "clicked_checkout", "metadata": {"source": "mobile"}},
    {"action": "abandoned_cart", "metadata": {"source": "ios"}},
]


# -----------------------------------------------------------------------
# Console state
# -----------------------------------------------------------------------

class Console:
    def __init__(self, db_path: str = "/tmp/quorex_console", dim: int = 8):
        self.db_path = db_path
        self.dim = dim
        self.encoder: Encoder | None = None
        self.engine: VectorDBEngine | None = None

    def boot(self, fresh: bool = False) -> None:
        if fresh and os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)
            step(f"wiped {self.db_path}")

        banner("Fitting encoder")
        self.encoder = Encoder(n_components=self.dim)
        self.encoder.fit(SEED_EVENTS)

        banner("Starting engine")
        self.engine = VectorDBEngine(
            path=self.db_path,
            dim=self.dim,
            M=4,
            ef_construction=20,
            ef_search=10,
            checkpoint_every=20,
            compact_after_deletes=5,
        )
        self.engine.start()
        self._show_stats()

    def shutdown(self) -> None:
        if self.engine:
            banner("Stopping engine")
            self.engine.stop()
            self.engine = None

    # ---------- commands ----------

    def cmd_insert(self, user: str, action: str, *kv: str) -> None:
        meta = self._parse_kv(kv)
        event = {"action": action, "metadata": meta}
        step(f"encoding event {event}")
        vec = self.encoder.encode(event)
        step(f"vector dim={vec.shape[0]} norm={float((vec*vec).sum())**0.5:.3f}")
        step("→ WAL.log_insert (with vector) — fsync")
        step("→ Segment.insert into HNSW graph")
        vec_id = self.engine.insert(user, vec, event)
        ok(f"inserted user={user} vec_id={vec_id}")
        self._show_stats()

    def cmd_search(self, user: str, *terms: str) -> None:
        query_str = " ".join(terms)
        event = {"action": query_str, "metadata": {}}
        step(f"encoding query: {query_str!r}")
        vec = self.encoder.encode(event)
        step("→ Segment.search (HNSW ANN)")
        results = self.engine.search(user, vec, top_k=5)
        if not results:
            warn("no results")
        else:
            for r in results:
                action = r["meta"].get("action", "?")
                print(f"  {C.MAGENTA}#{r['id']:>3}{C.RESET}  score={r['score']:.4f}  → {action}")

    def cmd_delete(self, user: str, vec_id: str) -> None:
        vid = int(vec_id)
        step(f"→ WAL.log_delete")
        step(f"→ HNSW.delete (detach from neighbors, possibly pick new entry_point)")
        ok_ = self.engine.delete(user, vid)
        if ok_:
            ok(f"deleted user={user} vec_id={vid}")
            pd = self.engine.segment.pending_deletes(user)
            step(f"pending_deletes for {user}: {pd}/{self.engine.compact_after_deletes}"
                 + (" — auto-compact triggered" if pd == 0 else ""))
        else:
            err(f"no such vector user={user} vec_id={vid}")
        self._show_stats()

    def cmd_update(self, user: str, vec_id: str, new_action: str, *kv: str) -> None:
        vid = int(vec_id)
        meta = self._parse_kv(kv)
        event = {"action": new_action, "metadata": meta}
        step(f"encoding new event {event}")
        vec = self.encoder.encode(event)
        step("→ WAL.log_update + HNSW.update (delete + reinsert)")
        ok_ = self.engine.update(user, vid, vec, event)
        if ok_:
            ok(f"updated user={user} vec_id={vid}")
        else:
            err(f"no such vector user={user} vec_id={vid}")

    def cmd_compact(self, user: str | None = None) -> None:
        step(f"→ Segment.compact{f' for {user}' if user else ' (all users)'}")
        step("rebuilds HNSW graph from live nodes — recovers ANN quality")
        n = self.engine.compact(user)
        ok(f"compacted {n} user(s)")

    def cmd_checkpoint(self) -> None:
        if not self.engine._dirty:
            warn("no changes since last checkpoint — would be a no-op (dirty flag)")
            return
        step("→ Storage.save (vectors.bin + meta.json + index.json + topology.json)")
        step("→ WAL.log_checkpoint + truncate_after_checkpoint")
        self.engine.checkpoint()
        self._show_stats()

    def cmd_learn(self, action: str, *kv: str) -> None:
        """partial_fit on a brand-new event — grow vocab online."""
        meta = self._parse_kv(kv)
        event = {"action": action, "metadata": meta}
        before = len(self.encoder.vectorizer.vocabulary)
        added = self.encoder.partial_fit([event])
        after = len(self.encoder.vectorizer.vocabulary)
        ok(f"vocab {before} → {after} (+{added} new tokens)")
        step("reducer.extend_vocab padded SVD components with zero columns")
        step("call `refit` to retrain SVD on the full corpus when ready")

    def cmd_refit(self) -> None:
        step("retraining SVD on accumulated corpus...")
        self.encoder.refit()
        ok("encoder refitted")

    def cmd_restart(self) -> None:
        """Stop + start — proves recovery works end-to-end."""
        banner("Restart (recovery test)")
        self.engine.stop()
        self.engine = VectorDBEngine(
            path=self.db_path,
            dim=self.dim,
            M=4,
            ef_construction=20,
            ef_search=10,
            checkpoint_every=20,
            compact_after_deletes=5,
        )
        self.engine.start()
        self._show_stats()

    def cmd_crash(self) -> None:
        """Simulate a crash: close the WAL handle WITHOUT checkpointing."""
        banner("Simulated crash (WAL closed, no checkpoint)")
        self.engine.wal.close()
        self.engine = None
        warn("engine reference dropped — type `restart` to recover from WAL")

    def cmd_stats(self) -> None:
        self._show_stats(verbose=True)

    def cmd_files(self) -> None:
        banner("Files on disk")
        if not os.path.exists(self.db_path):
            warn(f"{self.db_path} does not exist")
            return
        for root, _, files in os.walk(self.db_path):
            rel_root = os.path.relpath(root, self.db_path)
            prefix = "" if rel_root == "." else rel_root + "/"
            for f in sorted(files):
                full = os.path.join(root, f)
                size = os.path.getsize(full)
                print(f"  {prefix}{f:<20}  {size:>8} bytes")

    def cmd_wipe(self) -> None:
        warn(f"this will delete {self.db_path}")
        confirm = input("  type YES to confirm: ").strip()
        if confirm == "YES":
            self.shutdown()
            shutil.rmtree(self.db_path, ignore_errors=True)
            ok("wiped — booting fresh")
            self.boot(fresh=False)
        else:
            step("aborted")

    def cmd_help(self) -> None:
        banner("Commands")
        rows = [
            ("insert <user> <action> [k=v ...]",  "insert an event for a user"),
            ("search <user> <terms...>",          "ANN search"),
            ("delete <user> <vec_id>",            "hard delete a vector"),
            ("update <user> <vec_id> <action> [k=v ...]", "replace a vector + meta"),
            ("compact [user]",                    "rebuild HNSW graph (clean state)"),
            ("checkpoint",                        "force snapshot to disk"),
            ("learn <action> [k=v ...]",          "partial_fit encoder on new event"),
            ("refit",                             "retrain SVD on full corpus"),
            ("restart",                           "stop+start (test recovery)"),
            ("crash",                             "simulate crash (no checkpoint)"),
            ("stats",                             "show engine stats"),
            ("files",                             "list files on disk"),
            ("wipe",                              "delete the DB and reboot"),
            ("quit / exit",                       "stop and exit"),
        ]
        for cmd, desc in rows:
            print(f"  {C.BOLD}{cmd:<48}{C.RESET}{desc}")

    # ---------- internals ----------

    def _parse_kv(self, kv: tuple) -> dict:
        out = {}
        for token in kv:
            if "=" in token:
                k, v = token.split("=", 1)
                out[k.strip()] = v.strip()
        return out

    def _show_stats(self, verbose: bool = False) -> None:
        s = self.engine.stats()
        line = (
            f"  {C.DIM}users={s['users']}  vectors={s['total_vectors']}  "
            f"WAL={s['wal_size_bytes']}B  snapshot={s['storage_size_bytes']}B  "
            f"dirty={self.engine._dirty}{C.RESET}"
        )
        print(line)
        if verbose:
            for uid in self.engine.segment._indexes:
                n = self.engine.segment.vector_count(uid)
                pd = self.engine.segment.pending_deletes(uid)
                print(f"    · {uid}: {n} live vectors, {pd} pending deletes")


# -----------------------------------------------------------------------
# REPL
# -----------------------------------------------------------------------

DISPATCH = {
    "insert": "cmd_insert",
    "search": "cmd_search",
    "delete": "cmd_delete",
    "update": "cmd_update",
    "compact": "cmd_compact",
    "checkpoint": "cmd_checkpoint",
    "learn": "cmd_learn",
    "refit": "cmd_refit",
    "restart": "cmd_restart",
    "crash": "cmd_crash",
    "stats": "cmd_stats",
    "files": "cmd_files",
    "wipe": "cmd_wipe",
    "help": "cmd_help",
    "?": "cmd_help",
}


def repl():
    print(f"{C.BOLD}Quorex interactive console{C.RESET}")
    print(f"{C.DIM}Type `help` for commands, `quit` to exit.{C.RESET}")

    console = Console()
    console.boot(fresh=False)
    console.cmd_help()

    while True:
        try:
            raw = input(f"\n{C.CYAN}quorex>{C.RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            continue
        if raw in ("quit", "exit", "q"):
            break

        parts = raw.split()
        cmd, args = parts[0], parts[1:]

        method_name = DISPATCH.get(cmd)
        if not method_name:
            err(f"unknown command: {cmd!r} — type `help`")
            continue

        method = getattr(console, method_name)
        try:
            method(*args)
        except TypeError as e:
            err(f"bad arguments: {e}")
        except Exception as e:
            err(f"{type(e).__name__}: {e}")

    console.shutdown()
    print(f"\n{C.DIM}bye.{C.RESET}")


if __name__ == "__main__":
    repl()
