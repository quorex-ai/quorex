from pathlib import Path

import yaml
from alembic import op
import sqlalchemy as sa

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

EMBEDDING_DIM = 768 # intfloat / multilangual-e5-base ; voir tenants.embedding_dim

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE tenants (
    id              uuid PRIMARY KEY,
    name            text NOT NULL,
    embedding_model text NOT NULL,
    embedding_dim   integer NOT NULL,
    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE api_keys (
    id          uuid PRIMARY KEY,
    tenant_id   uuid NOT NULL REFERENCES tenants(id),
    key_hash    text NOT NULL UNIQUE,
    key_prefix  text NOT NULL,
    label       text,
    created_at  timestamptz NOT NULL DEFAULT now(),
    revoked_at  timestamptz
);
CREATE INDEX api_keys_tenant_idx ON api_keys(tenant_id);
CREATE INDEX api_keys_prefix_idx ON api_keys(key_prefix) WHERE revoked_at IS NULL;

CREATE TABLE users (
    id          uuid PRIMARY KEY,
    tenant_id   uuid NOT NULL REFERENCES tenants(id),
    external_id text NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, external_id)
);

CREATE TABLE messages (
    id               uuid PRIMARY KEY,
    tenant_id        uuid NOT NULL REFERENCES tenants(id),
    user_id          uuid NOT NULL REFERENCES users(id),
    content          text NOT NULL,
    received_at      timestamptz NOT NULL DEFAULT now(),
    extraction_model text,
    extraction_raw   jsonb
);
CREATE INDEX messages_user_received_idx ON messages(user_id, received_at DESC);

CREATE TYPE durability AS ENUM ('permanent', 'durable', 'transient');
CREATE TYPE end_reason AS ENUM ('replaced', 'invalidated', 'expired', 'forgotten');

CREATE TABLE facts (
    id                  uuid PRIMARY KEY,
    tenant_id           uuid NOT NULL REFERENCES tenants(id),
    user_id             uuid NOT NULL REFERENCES users(id),
    subject             text NOT NULL DEFAULT 'user',
    attribute           text NOT NULL,
    attribute_raw       text NOT NULL,
    value               text NOT NULL,
    confidence          real NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    durability          durability NOT NULL,
    expires_at          timestamptz,
    source_message_id   uuid REFERENCES messages(id),
    valid_from          timestamptz NOT NULL,
    valid_to            timestamptz,
    recorded_at         timestamptz NOT NULL DEFAULT now(),
    replaced_by         uuid REFERENCES facts(id),
    end_reason          end_reason,
    attribute_embedding vector(768),
    value_embedding     vector(768),
    CONSTRAINT facts_closed_has_reason   CHECK ((valid_to IS NULL) = (end_reason IS NULL)),
    CONSTRAINT facts_replaced_by_reason  CHECK (replaced_by IS NULL OR end_reason = 'replaced'),
    CONSTRAINT facts_transient_expires   CHECK (durability <> 'transient' OR expires_at IS NOT NULL),
    CONSTRAINT facts_valid_range         CHECK (valid_to IS NULL OR valid_to >= valid_from)
);

CREATE UNIQUE INDEX facts_active_unique
    ON facts(user_id, subject, attribute) WHERE valid_to IS NULL;
CREATE INDEX facts_user_active_idx ON facts(user_id) WHERE valid_to IS NULL;
CREATE INDEX facts_user_time_idx ON facts(user_id, recorded_at, valid_from, valid_to);
CREATE INDEX facts_expiry_idx ON facts(expires_at)
    WHERE valid_to IS NULL AND expires_at IS NOT NULL;
CREATE INDEX facts_attr_embedding_idx ON facts
    USING hnsw (attribute_embedding vector_cosine_ops);

CREATE TABLE synonyms (
    variant   text PRIMARY KEY,
    canonical text NOT NULL
);
CREATE INDEX synonyms_canonical_idx ON synonyms(canonical);
"""

def _load_synonyms() -> list[dict[str, str]]:
    path = Path(__file__).resolve().parents[2] / "data" / "synonyms.yaml"
    data = yaml.safe_load(path.read_text(encodings="utf-8")) or {}
    rows = list[dict[str, str]] = []
    seen: dict[str, str] = {}
    for canonical, variants in data.items():
        for variant in variants or []:
            if variant in seen and seen[variant] != canonical:
                raise ValueError(
                    f"Synonyms.yaml : '{variant}' mappé vers '{seen[variant]}' et '{canonical}'"
                )
            seen[variant] = canonical
            rows.append({"variant": variant, "canonical": canonical})
            return rows

def upgrade() -> None:
    op.execute(SCHEMA)
    rows = _load_synonyms()
    if rows:
        synonyms = sa.table(
            "synonyms", sa.column("variant", sa.Text), sa.column("canonical", sa.Text)
        )
        op.bulk_insert(synonyms, rows)

def downgrade() -> None:
    op.execute(
        """
        DROP TABLE IF EXISTS synonyms;
        DROP TABLE IF EXISTS facts;
        DROP TYPE IF EXISTS end_reason;
        DROP TYPE IF EXISTS durability;
        DROP TABLE IF EXISTS messages;
        DROP TABLE IF EXISTS users;
        DROP TABLE IF EXISTS api_keys;
        DROP TABLE IF EXISTS tenants;
        """
    )