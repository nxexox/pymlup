from mlup.utils.crypto import generate_unique_id


def test_generate_unique_id():
    count_checks = 10
    assert len({generate_unique_id() for _ in range(count_checks)}) == count_checks
