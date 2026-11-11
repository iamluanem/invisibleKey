import pytest

from invisiblekey.config import LayoutConfig
from invisiblekey.layout import SpecialKey, VirtualKeyboardLayout, shifted


@pytest.fixture
def layout():
    return VirtualKeyboardLayout()


@pytest.mark.parametrize("key_id", ["q", "a", "z", "1", "0", "p", ";", "/", "space", "enter", "backspace", "shift_l"])
def test_key_center_hits_itself(layout, key_id):
    hit = layout.hit_test(*layout.key_center_surface(key_id))
    assert hit.key.id == key_id and hit.inside and hit.centrality == pytest.approx(1.0)


def test_required_keys_present(layout):
    ids = {k.id for k in layout.keys}
    assert set("abcdefghijklmnopqrstuvwxyz0123456789,.;'/-=") <= ids
    assert {k.special for k in layout.keys if k.special} == set(SpecialKey)


def test_keys_do_not_overlap(layout):
    for i, a in enumerate(layout.keys):
        for b in layout.keys[i + 1:]:
            overlap_x = min(a.x + a.w, b.x + b.w) - max(a.x, b.x)
            overlap_y = min(a.y + a.h, b.y + b.h) - max(a.y, b.y)
            assert overlap_x <= 1e-9 or overlap_y <= 1e-9, (a.id, b.id)


def test_edge_hit_has_lower_centrality(layout):
    k = layout.key("g")
    u, v = layout.to_surface(k.x + 0.95, k.y + 0.5)
    hit = layout.hit_test(u, v)
    assert hit.key.id == "g" and 0.5 <= hit.centrality < 0.6


def test_tolerance_outside_keys(layout):
    # Left of "q" (no Tab key there): nearest key within tolerance, low centrality.
    u, v = layout.to_surface(1.3, 1.5)
    hit = layout.hit_test(u, v)
    assert hit.key.id == "q" and not hit.inside and hit.centrality < 0.5
    assert layout.hit_test(*layout.to_surface(-2, 2)) is None


def test_layout_scale_and_offset():
    lay = VirtualKeyboardLayout(LayoutConfig(x=0.1, y=0.2, width=0.5, height=0.5))
    u, v = lay.key_center_surface("space")
    assert 0.1 < u < 0.6 and 0.2 < v < 0.7
    assert lay.hit_test(u, v).key.id == "space"
    assert lay.hit_test(0.9, 0.9) is None


def test_shift_mapping():
    assert shifted("a") == "A" and shifted("1") == "!" and shifted("/") == "?" and shifted(",") == "<"
