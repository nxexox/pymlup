import uuid


def generate_unique_id() -> str:
    return str(uuid.uuid4())
