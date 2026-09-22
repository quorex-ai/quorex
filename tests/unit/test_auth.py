from quorex.auth import extract_prefix, generate_api_key, verify_api_key


def test_generation_et_verification() -> None:
    k = generate_api_key()
    assert k.plain.startswith("qx_") and len(k.plain) == 3 + 8 + 32
    assert extract_prefix(k.plain) == k.prefix
    assert verify_api_key(k.plain, k.hash)
    assert not verify_api_key(k.plain[:-1] + "x", k.hash)


def test_prefix_refuse_les_formats_invalides() -> None:
    assert extract_prefix("") is None
    assert extract_prefix("qx_court") is None
    assert extract_prefix("sk_" + "a" * 40) is None
    assert extract_prefix("qx_" + "a" * 39 + "!") is None