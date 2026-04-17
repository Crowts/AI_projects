import pytest
from src.multi_agent import add, multiply

def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
    assert add(0.1, 0.2) == pytest.approx(0.3, rel=1e-6)  # Handle floating-point precision
    assert add(-5, -5) == -10

    with pytest.raises(TypeError):
        add("2", 3)
    with pytest.raises(TypeError):
        add(None, 3)

def test_multiply():
    assert multiply(2, 3) == 6
    assert multiply(-1, 1) == -1
    assert multiply(0.1, 0.2) == pytest.approx(0.02, rel=1e-6)
    assert multiply(-5, -5) == 25
    assert multiply(0, 5) == 0
    
    with pytest.raises(TypeError):
        multiply("2", 3)
    with pytest.raises(TypeError):
        multiply(None, 3)
