import pytest


def f():
    raise SystemExit(1)


def test_mytest():
    with pytest.raises(SystemExit):
        f()


def add_two(x):
    return x + 2


def test_testAdd():
    assert add_two(3) == 5
