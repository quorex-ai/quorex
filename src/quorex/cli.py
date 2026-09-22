from __future__ import annotations

from uuid import UUID

import typer
import uvicorn

from quorex.auth import generate_api_key
from quorex.settings import get_settings
from quorex.storage import Database, PostgresTenantStore

app = typer.Typer(no_args_is_help=True, help="QUOREX, mémoire de faits pour agents IA.")
tenant_app = typer.Typer(no_args_is_help=True, help="Gestion des tenants.")
key_app = typer.Typer(no_args_is_help=True, help="Gestion des clés API.")
app.add_typer(tenant_app, name="tenant")
app.add_typer(key_app, name="key")

def _tenants() -> PostgresTenantStore:
    return PostgresTenantStore(Database(get_settings().database_url))

@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    """Lance l'API QUOREX."""
    uvicorn.run("quorex.api:create_app", factory=True, host=host, port=port, reload=reload)

@tenant_app.command("create")
def tenant_create(name: str, label: str = "default") -> None:
    """Créer un tenant et sa première clé API (affichée une seule fois)."""
    settings = get_settings()
    store = _tenants()
    tenant = store.create_tenant(name, settings.embedding_model, 768)
    key = generate_api_key()
    store.create_api_key(tenant.id, key.prefix, key.hash, label)
    typer.echo(f"tenant_id : {tenant.id}")
    typer.echo(f"api_key : {key.plain}")
    typer.echo("Copie la clé maintenant, elle ne sera jamais réaffichée.")

@key_app.command("create")
def key_create(tenant_id: UUID, label: str = "default") -> None:
    """Crée une clé supplémentaire pour un tenant existant."""
    store = _tenants()
    if store.get_tenant(tenant_id) is None:
        raise typer.BadParameter("tenant inconnu")
    key = generate_api_key()
    store.create_api_key(tenant_id, key.prefix, key.hash, label)
    typer.echo(f"api_key : {key.plain}")

@key_app.command("revoke")
def key_revoke(key_id: UUID) -> None:
    """Révoque une clé. Effet immédiat."""
    _tenants().revoke_api_key(key_id)
    typer.echo("révoquée")