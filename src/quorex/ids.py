from uuid import UUID

import uuid_utils

def new_id() -> UUID:
    return UUID(bytes=uuid_utils.uuid7().bytes)