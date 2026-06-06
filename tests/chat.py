"""
Chatbot interface for the Quorex vector DB.

Type freely. Every message is encoded, stored as a memory, and matched
against past memories — the bot replies with the closest things it
remembers. Vocabulary grows online as new words appear.

Slash commands:
    /help        list commands
    /stats       memory + storage stats
    /forget <n>  delete memory #n
    /refit       retrain SVD on the full corpus (sharpens new words)
    /restart     close + reopen the DB (proves recovery)
    /wipe        delete everything and start over
    /quit        exit

Run:
    python3 tests/chat.py
"""

from __future__ import annotations

import os
import sys
import shutil
import time

# Make src/ importable without PYTHONPATH or pip install.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quorex.core.embeddings.encoder import Encoder
from quorex.core.vectordb.engine import VectorDBEngine


DB_PATH = "/tmp/quorex_chat"
USER_ID = "me"
TARGET_DIM = 32  # target — may shrink if the seed corpus is smaller


# A seed corpus large and overlapping enough that the SVD learns
# meaningful co-occurrence structure (not pure one-hot orthogonality).
SEED_CORPUS = [
    # Greetings FR
    {"action": "coucou bonjour salut hello bonsoir", "metadata": {"context": "greeting"}},
    {"action": "bonjour comment allez vous", "metadata": {"context": "greeting"}},
    {"action": "salut ca va bien merci", "metadata": {"context": "greeting"}},
    {"action": "bonsoir bonne nuit aurevoir", "metadata": {"context": "greeting"}},
    {"action": "hey yo wesh quoi de neuf", "metadata": {"context": "greeting"}},
    {"action": "coucou comment tu vas toi", "metadata": {"context": "greeting"}},
    {"action": "bonjour monde entier ici", "metadata": {"context": "greeting"}},
    {"action": "salut tout le monde", "metadata": {"context": "greeting"}},
 
    # Responses FR
    {"action": "oui non peut etre absolument", "metadata": {"context": "response"}},
    {"action": "ok parfait super bien excellent", "metadata": {"context": "positive"}},
    {"action": "non pas du tout jamais rien", "metadata": {"context": "negative"}},
    {"action": "je sais pas comprends rien", "metadata": {"context": "confusion"}},
    {"action": "peut etre bof mouais moyen", "metadata": {"context": "neutral"}},
    {"action": "evidemment bien sur forcement", "metadata": {"context": "affirmation"}},
    {"action": "vraiment serieusement franchement", "metadata": {"context": "emphasis"}},
    {"action": "exactement tout fait accord", "metadata": {"context": "agreement"}},
 
    # Emotions FR
    {"action": "triste mal pas bien deprime", "metadata": {"context": "negative_emotion"}},
    {"action": "content heureux super joyeux", "metadata": {"context": "positive_emotion"}},
    {"action": "fatigue epuise nul sommeil", "metadata": {"context": "tired"}},
    {"action": "stresse angoisse peur inquiet", "metadata": {"context": "stress"}},
    {"action": "amour aime adore passion", "metadata": {"context": "love"}},
    {"action": "colere fache energie fort", "metadata": {"context": "anger"}},
    {"action": "calme tranquille repos paisible", "metadata": {"context": "calm"}},
    {"action": "excite impatient hate vivement", "metadata": {"context": "excitement"}},
    {"action": "jaloux envieux frustre decu", "metadata": {"context": "frustration"}},
    {"action": "fier satisfait accompli reussi", "metadata": {"context": "pride"}},
 
    # Daily life FR
    {"action": "manger boire faim soif repas", "metadata": {"context": "food"}},
    {"action": "dormir nuit sommeil reveille", "metadata": {"context": "sleep"}},
    {"action": "travailler bureau job boulot", "metadata": {"context": "work"}},
    {"action": "etudier cours ecole universite", "metadata": {"context": "education"}},
    {"action": "sortir amis soiree fete", "metadata": {"context": "social"}},
    {"action": "sport gym course musculation", "metadata": {"context": "sport"}},
    {"action": "film serie television regarder", "metadata": {"context": "entertainment"}},
    {"action": "musique ecouter chanson concert", "metadata": {"context": "music"}},
    {"action": "lire livre lecture roman", "metadata": {"context": "reading"}},
    {"action": "voyager vacances avion hotel", "metadata": {"context": "travel"}},
    {"action": "conduire voiture route trajet", "metadata": {"context": "transport"}},
    {"action": "cuisiner diner recette plat", "metadata": {"context": "cooking"}},
    {"action": "acheter courses marche magasin", "metadata": {"context": "shopping"}},
    {"action": "telephone appel message texte", "metadata": {"context": "communication"}},
    {"action": "ordinateur internet naviguer chercher", "metadata": {"context": "tech"}},
 
    # Tech FR
    {"action": "code programmer developer application", "metadata": {"context": "tech"}},
    {"action": "bug erreur probleme fixer", "metadata": {"context": "tech"}},
    {"action": "api serveur backend frontend", "metadata": {"context": "tech"}},
    {"action": "base donnees sql requete", "metadata": {"context": "tech"}},
    {"action": "intelligence artificielle modele llm", "metadata": {"context": "ai"}},
    {"action": "github git commit push pull", "metadata": {"context": "tech"}},
    {"action": "deployer production cloud serveur", "metadata": {"context": "tech"}},
    {"action": "python javascript typescript react", "metadata": {"context": "tech"}},
    {"action": "tester unittest debug performance", "metadata": {"context": "tech"}},
    {"action": "projet startup idee produit", "metadata": {"context": "business"}},
 
    # Questions FR
    {"action": "pourquoi comment quand ou qui", "metadata": {"context": "question"}},
    {"action": "quest ce que signifie veut dire", "metadata": {"context": "question"}},
    {"action": "peux tu aide besoin help", "metadata": {"context": "request"}},
    {"action": "que penses tu avis opinion", "metadata": {"context": "opinion"}},
    {"action": "connais sais informations details", "metadata": {"context": "knowledge"}},
    {"action": "explique montre donne exemple", "metadata": {"context": "request"}},
 
    # Time FR
    {"action": "aujourd hui hier demain maintenant", "metadata": {"context": "time"}},
    {"action": "matin midi soir nuit minuit", "metadata": {"context": "time"}},
    {"action": "semaine mois annee date heure", "metadata": {"context": "time"}},
    {"action": "bientot jamais toujours parfois souvent", "metadata": {"context": "frequency"}},
    {"action": "avant apres pendant depuis longtemps", "metadata": {"context": "time"}},
 
    # Greetings EN
    {"action": "hello hi hey good morning", "metadata": {"context": "greeting"}},
    {"action": "good evening good night bye", "metadata": {"context": "greeting"}},
    {"action": "how are you doing fine thanks", "metadata": {"context": "greeting"}},
    {"action": "what is up nothing much cool", "metadata": {"context": "greeting"}},
    {"action": "nice to meet you pleased", "metadata": {"context": "greeting"}},
 
    # Responses EN
    {"action": "yes no maybe sure absolutely", "metadata": {"context": "response"}},
    {"action": "ok great perfect awesome excellent", "metadata": {"context": "positive"}},
    {"action": "no way never not at all", "metadata": {"context": "negative"}},
    {"action": "i don not know unclear confused", "metadata": {"context": "confusion"}},
    {"action": "obviously of course definitely", "metadata": {"context": "affirmation"}},
    {"action": "exactly right agree totally", "metadata": {"context": "agreement"}},
    {"action": "really seriously honestly truly", "metadata": {"context": "emphasis"}},
    {"action": "kind of sort of not really", "metadata": {"context": "neutral"}},
 
    # Emotions EN
    {"action": "sad bad feeling down depressed", "metadata": {"context": "negative_emotion"}},
    {"action": "happy glad excited joyful", "metadata": {"context": "positive_emotion"}},
    {"action": "tired exhausted sleepy drained", "metadata": {"context": "tired"}},
    {"action": "stressed anxious worried nervous", "metadata": {"context": "stress"}},
    {"action": "love like enjoy appreciate", "metadata": {"context": "love"}},
    {"action": "angry mad frustrated upset", "metadata": {"context": "anger"}},
    {"action": "calm relaxed peaceful chill", "metadata": {"context": "calm"}},
    {"action": "proud satisfied accomplished done", "metadata": {"context": "pride"}},
    {"action": "bored lonely empty nothing", "metadata": {"context": "boredom"}},
    {"action": "grateful thankful blessed lucky", "metadata": {"context": "gratitude"}},
 
    # Daily EN
    {"action": "eat drink food hungry thirsty", "metadata": {"context": "food"}},
    {"action": "sleep wake rest bed night", "metadata": {"context": "sleep"}},
    {"action": "work job office meeting", "metadata": {"context": "work"}},
    {"action": "study learn school university", "metadata": {"context": "education"}},
    {"action": "go out friends party social", "metadata": {"context": "social"}},
    {"action": "exercise run gym fitness", "metadata": {"context": "sport"}},
    {"action": "watch movie series tv show", "metadata": {"context": "entertainment"}},
    {"action": "listen music song playlist", "metadata": {"context": "music"}},
    {"action": "read book novel story", "metadata": {"context": "reading"}},
    {"action": "travel trip flight hotel", "metadata": {"context": "travel"}},
    {"action": "drive car road commute", "metadata": {"context": "transport"}},
    {"action": "cook dinner recipe meal", "metadata": {"context": "cooking"}},
    {"action": "buy shop market store", "metadata": {"context": "shopping"}},
    {"action": "call text message phone", "metadata": {"context": "communication"}},
    {"action": "browse internet search online", "metadata": {"context": "tech"}},
 
    # Tech EN
    {"action": "code program build deploy", "metadata": {"context": "tech"}},
    {"action": "bug fix error debug issue", "metadata": {"context": "tech"}},
    {"action": "api server database query", "metadata": {"context": "tech"}},
    {"action": "machine learning model train", "metadata": {"context": "ai"}},
    {"action": "startup product launch idea", "metadata": {"context": "business"}},
    {"action": "design figma ui ux interface", "metadata": {"context": "design"}},
    {"action": "test performance optimize scale", "metadata": {"context": "tech"}},
 
    # Questions EN
    {"action": "why how when where who what", "metadata": {"context": "question"}},
    {"action": "can you help me please need", "metadata": {"context": "request"}},
    {"action": "what do you think opinion view", "metadata": {"context": "opinion"}},
    {"action": "do you know information about", "metadata": {"context": "knowledge"}},
    {"action": "explain show give example tell", "metadata": {"context": "request"}},
 
    # Time EN
    {"action": "today yesterday tomorrow now", "metadata": {"context": "time"}},
    {"action": "morning noon evening night", "metadata": {"context": "time"}},
    {"action": "week month year date time", "metadata": {"context": "time"}},
    {"action": "soon never always sometimes often", "metadata": {"context": "frequency"}},
    {"action": "before after during since long", "metadata": {"context": "time"}},
 
    # Misc common words
    {"action": "je tu il elle nous vous ils", "metadata": {"context": "pronouns"}},
    {"action": "mon ma mes ton ta tes son sa", "metadata": {"context": "possessives"}},
    {"action": "le la les un une des", "metadata": {"context": "articles"}},
    {"action": "et ou mais donc car or ni", "metadata": {"context": "conjunctions"}},
    {"action": "tres bien mal peu beaucoup", "metadata": {"context": "adverbs"}},
    {"action": "grand petit vite lent haut bas", "metadata": {"context": "adjectives"}},
    {"action": "faire aller venir voir dire", "metadata": {"context": "verbs"}},
    {"action": "avoir etre vouloir pouvoir savoir", "metadata": {"context": "verbs"}},
    {"action": "the a an is are was were", "metadata": {"context": "english_basics"}},
    {"action": "i you he she we they it", "metadata": {"context": "pronouns"}},
    {"action": "my your his her our their", "metadata": {"context": "possessives"}},
    {"action": "and or but so because", "metadata": {"context": "conjunctions"}},
    {"action": "very much little fast slow", "metadata": {"context": "adverbs"}},
    {"action": "have be want can know go", "metadata": {"context": "verbs"}},
    {"action": "good bad big small new old", "metadata": {"context": "adjectives"}},
 
    # Numbers + common
    {"action": "un deux trois quatre cinq six", "metadata": {"context": "numbers"}},
    {"action": "one two three four five six", "metadata": {"context": "numbers"}},
    {"action": "premier dernier prochain suivant", "metadata": {"context": "ordinals"}},
    {"action": "first last next previous same", "metadata": {"context": "ordinals"}},
 
    # Pricing / product (original)
    {"action": "viewed pricing page plan pro", "metadata": {"context": "product"}},
    {"action": "searched pricing compared plans", "metadata": {"context": "product"}},
    {"action": "upgraded plan billing subscription", "metadata": {"context": "product"}},
    {"action": "clicked checkout cta dashboard", "metadata": {"context": "product"}},
    {"action": "visited homepage organic traffic", "metadata": {"context": "product"}},
    {"action": "searched docs api reference", "metadata": {"context": "product"}},
    {"action": "opened onboarding email welcome", "metadata": {"context": "product"}},
    {"action": "viewed integrations marketplace", "metadata": {"context": "product"}},
]


class C:
    RESET = "\033[0m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    MAGENTA = "\033[35m"


def bot_say(text: str) -> None:
    print(f"{C.CYAN}bot>{C.RESET} {text}")


def meta(text: str) -> None:
    print(f"     {C.DIM}{text}{C.RESET}")


# -----------------------------------------------------------------------
# Chat
# -----------------------------------------------------------------------

class Chat:
    def __init__(self):
        self.encoder = Encoder(n_components=TARGET_DIM)
        print(f"{C.DIM}fitting encoder on seed corpus ({len(SEED_CORPUS)} events)...{C.RESET}")
        self.encoder.fit(SEED_CORPUS)

        # SVDReducer silently caps n_components to corpus size. Use the
        # ACTUAL post-fit dim for the engine so storage round-trips work.
        self.dim = self.encoder.reducer.n_components
        if self.dim != TARGET_DIM:
            print(f"{C.DIM}note: SVD shrank dim {TARGET_DIM} → {self.dim} (corpus too small).{C.RESET}")

        self.engine = self._make_engine()
        self.engine.start()

        # If the snapshot loaded uses a different dim, we have a problem.
        # Wipe + warn so the chat can keep running rather than crash later.
        if self.engine.segment.total_vectors() == 0:
            return
        sample_user = next(iter(self.engine.segment._indexes), None)
        if sample_user:
            sample_id = next(iter(self.engine.segment._indexes[sample_user].nodes), None)
            if sample_id is not None:
                saved_dim = self.engine.segment.get_vector(sample_user, sample_id).shape[0]
                if saved_dim != self.dim:
                    print(f"{C.DIM}note: on-disk dim ({saved_dim}) ≠ current ({self.dim}) — wiping stale snapshot.{C.RESET}")
                    self.engine.stop()
                    shutil.rmtree(DB_PATH, ignore_errors=True)
                    self.engine = self._make_engine()
                    self.engine.start()

    def _make_engine(self) -> VectorDBEngine:
        return VectorDBEngine(
            path=DB_PATH,
            dim=self.dim,
            M=8,
            ef_construction=40,
            ef_search=20,
            checkpoint_every=10,
            compact_after_deletes=5,
        )

    def stop(self) -> None:
        self.engine.stop()


    def _rebuild_index(self) -> None:
        """
        Re-encoder all stored memories with the updated encoder.
        Called after refit() so HNSW vectors stay in sync with the new SVD.
        """
        seg = self.engine.segment
        if USER_ID not in seg._metadata:
            return
        for vec_id, stored_meta in seg._metadata[USER_ID].items():
            # Extract the original event - strip userId added by engine
            event = {
                "action": stored_meta.get("action", "message"),
                "metadata": stored_meta.get("metadata", {})
            }
            new_vec = self.encoder.encode(event)
            if vec_id in seg._indexes[USER_ID].nodes:
                seg._indexes[USER_ID].nodes[vec_id].vector = \
                seg._indexes[USER_ID]._normalize(new_vec)
    # ---------- core loop ----------

    def handle(self, text: str) -> None:
        if text.startswith("/"):
            self.handle_command(text[1:].strip())
            return
        if not text.strip():
            return

        # Wrap free-text input as a dict event so the encoder can chew it.
        event = {
            "action": text,
            "metadata": {
                "text": text,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        }

        # 1 — Online vocab growth. The reducer assigns each new token a
        # deterministic hashed projection column, so new words contribute
        # real signal immediately. No refit needed for correctness; call
        # /refit only to sharpen quality after many new tokens.
        added = self.encoder.partial_fit([event])

        # 2 — Encode + search BEFORE insert so we don't trivially match
        # against ourselves.
        query_vec = self.encoder.encode(event)
        print(f"     DEBUG vec: {query_vec.round(3)}")

        if USER_ID in self.engine.segment._indexes:
            idx = self.engine.segment._indexes[USER_ID]
            for nid, node in list(idx.nodes.items())[:3]:
                d = idx._dist(query_vec, node.vector)
                print(f"     DEBUG dist to #{nid}: {d:.4f} -> sim={1-d:.4f}")

        hits = self.engine.search(USER_ID, query_vec, top_k=3, threshold=-0.7)

        # 3 — Store the new memory.
        vec_id = self.engine.insert(USER_ID, query_vec, event)

        # 4 — Bot reply.
        if added:
            meta(f"learned {added} new word(s) — vocab now {len(self.encoder.vectorizer.vocabulary)}")
        meta(f"stored as memory #{vec_id}")

        if not hits:
            bot_say("got it. that's new territory — nothing similar in memory yet.")
            return

        if len(hits) == 1 and hits[0]["score"] > 0.99:
            # the only "hit" is the one we just inserted — but we searched BEFORE insert
            # so this shouldn't happen. Defensive.
            bot_say("noted.")
            return

        bot_say(f"that reminds me of {len(hits)} thing(s):")
        for h in hits:
            past_text = h["meta"].get("metadata", {}).get("text", "(no text)")
            short = past_text if len(past_text) <= 60 else past_text[:57] + "..."
            print(f"     {C.MAGENTA}#{h['id']:<3}{C.RESET} {C.DIM}sim={h['score']:.3f}{C.RESET}  {short}")

    # ---------- slash commands ----------

    def handle_command(self, cmd: str) -> None:
        parts = cmd.split()
        if not parts:
            return
        name, args = parts[0], parts[1:]

        if name in ("quit", "exit", "q"):
            raise SystemExit
        if name == "help":
            self.cmd_help()
        elif name == "stats":
            self.cmd_stats()
        elif name == "forget":
            self.cmd_forget(args)
        elif name == "refit":
            self.cmd_refit()
        elif name == "restart":
            self.cmd_restart()
        elif name == "wipe":
            self.cmd_wipe()
        elif name == "list":
            self.cmd_list()
        else:
            bot_say(f"unknown command /{name}. try /help.")

    def cmd_help(self) -> None:
        bot_say("commands:")
        for line in [
            "/help          show this",
            "/stats         memory + storage stats",
            "/list          list all stored memories",
            "/forget <n>    delete memory #n",
            "/refit         retrain SVD on accumulated corpus",
            "/restart       stop + start the DB (recovery test)",
            "/wipe          delete everything and start fresh",
            "/quit          exit",
        ]:
            print(f"     {C.DIM}{line}{C.RESET}")

    def cmd_stats(self) -> None:
        s = self.engine.stats()
        vocab = len(self.encoder.vectorizer.vocabulary)
        bot_say(
            f"memories={s['total_vectors']} · vocab={vocab} tokens · "
            f"wal={s['wal_size_bytes']}B · snapshot={s['storage_size_bytes']}B"
        )

    def cmd_list(self) -> None:
        seg = self.engine.segment
        if USER_ID not in seg._metadata or not seg._metadata[USER_ID]:
            bot_say("no memories yet.")
            return
        bot_say("all memories:")
        for vid in sorted(seg._metadata[USER_ID].keys()):
            text = seg._metadata[USER_ID][vid].get("metadata", {}).get("text", "(no text)")
            short = text if len(text) <= 60 else text[:57] + "..."
            print(f"     {C.MAGENTA}#{vid:<3}{C.RESET}  {short}")

    def cmd_forget(self, args: list[str]) -> None:
        if not args:
            bot_say("usage: /forget <memory_id>")
            return
        try:
            vid = int(args[0])
        except ValueError:
            bot_say(f"not a number: {args[0]!r}")
            return
        ok = self.engine.delete(USER_ID, vid)
        if ok:
            bot_say(f"forgot memory #{vid}.")
        else:
            bot_say(f"no memory #{vid} to forget.")

    def cmd_refit(self) -> None:
        if not self.encoder._training_events:
            bot_say("nothing to refit on yet.")
            return
        meta(f"retraining SVD on {len(self.encoder._training_events)} events...")
        self.encoder.refit()
        bot_say("done — new words are now properly embedded.")

    def cmd_restart(self) -> None:
        meta("stopping engine + checkpointing (if dirty)...")
        self.engine.stop()
        meta("reopening from disk...")
        self.engine = self._make_engine()
        self.engine.start()
        s = self.engine.stats()
        bot_say(f"back — {s['total_vectors']} memories restored from disk.")

    def cmd_wipe(self) -> None:
        meta("wiping everything...")
        self.engine.stop()
        shutil.rmtree(DB_PATH, ignore_errors=True)
        self.engine = self._make_engine()
        self.engine.start()
        bot_say("fresh slate.")


# -----------------------------------------------------------------------
# REPL
# -----------------------------------------------------------------------

def main():
    print(f"{C.BOLD}quorex chatbot{C.RESET}")
    print(f"{C.DIM}type anything — i'll remember it and surface similar past inputs.{C.RESET}")
    print(f"{C.DIM}slash commands: /help /stats /list /forget /refit /restart /wipe /quit{C.RESET}\n")

    chat = Chat()
    bot_say("hi. tell me anything.")

    try:
        while True:
            try:
                raw = input(f"\n{C.GREEN}you>{C.RESET} ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not raw:
                continue
            try:
                chat.handle(raw)
            except SystemExit:
                break
            except Exception as e:
                print(f"     {C.YELLOW}!{C.RESET} {type(e).__name__}: {e}")
    finally:
        chat.stop()
        print(f"\n{C.DIM}bye.{C.RESET}")


if __name__ == "__main__":
    main()
