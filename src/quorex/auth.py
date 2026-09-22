from __future__ import annotations

import secrets
import string
from dataclasses import dataclass

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

_ALPHABET = string.ascii_letters + string.digits
_PREFIX_LEN = 8
_SECRET_LEN = 32
_HEADER = "qx_"

_hasher = PasswordHasher()

@dataclass(frozen=True, slots=True)
class GeneratedKey:
    plain: str # à montrer une seule fois à l'utilisateur
    prefix: str # stocké en clair, sert de clé de recherche
    hash: str # argon2id du plain, stocké

def generate_api_key() -> GeneratedKey:
    prefix = "".join(secrets.choice(_ALPHABET) for _ in range(_PREFIX_LEN))
    secret = "".join(secrets.choice(_ALPHABET) for _ in range(_SECRET_LEN))
    plain = f"{_HEADER}{prefix}{secret}"
    return GeneratedKey(plain=plain, prefix=prefix, hash=_hasher.hash(plain))

def extract_prefix(plain: str) -> str:
    """Retourne le préfixe d'une clé bien formée, None sinon."""
    if not plain.startswith(_HEADER):
        return None
    body = plain[len(_HEADER):]
    if len(body) != _PREFIX_LEN + _SECRET_LEN or not all(c in _ALPHABET for c in body):
        return None
    return body[:_PREFIX_LEN]

def verify_api_key(plain: str, key_hash: str) -> bool:
    try:
        return _hasher.verify(key_hash, plain)
    except VerifyMismatchError:
        return False 
    except Exception: # hash corrompu ou format inconnu : on refuse
        return False