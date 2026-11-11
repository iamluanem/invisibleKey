import numpy as np
import pytest

from invisiblekey.calibration import DeskCalibration, DeskCalibrationService
from invisiblekey.geometry import UNIT_SQUARE, CoordinateTransformer
from invisiblekey.settings import AppSettings, SettingsManager
from invisiblekey.simulation import DEFAULT_CORNERS


def test_corners_map_to_unit_square():
    tf = CoordinateTransformer(DEFAULT_CORNERS)
    np.testing.assert_allclose(tf.to_surface(DEFAULT_CORNERS), UNIT_SQUARE, atol=1e-9)


def test_round_trip_and_perspective_center():
    tf = CoordinateTransformer(DEFAULT_CORNERS)
    pts = np.random.default_rng(1).random((50, 2))
    np.testing.assert_allclose(tf.to_surface(tf.to_image(pts)), pts, atol=1e-9)
    # Under perspective the far half looks compressed: the center sits above the midpoint.
    center_img = tf.to_image([0.5, 0.5])[0]
    assert center_img[1] < np.mean(np.array(DEFAULT_CORNERS)[:, 1])


def test_identity_calibration():
    tf = CoordinateTransformer(UNIT_SQUARE)
    np.testing.assert_allclose(tf.to_surface([[0.3, 0.7]]), [[0.3, 0.7]], atol=1e-9)


@pytest.mark.parametrize("corners", [
    [(0, 0), (1, 0), (0, 1), (1, 1)],          # wrong order (self-intersecting)
    [(0.5, 0.5), (0.51, 0.5), (0.51, 0.51), (0.5, 0.51)],  # too small
    [(0, 0), (1, 0), (1, 1)],                  # 3 corners
])
def test_invalid_quads_rejected(corners):
    with pytest.raises(ValueError):
        CoordinateTransformer(corners)


def test_wizard_flow():
    svc = DeskCalibrationService()
    assert svc.next_corner == "top_left"
    for x, y in DEFAULT_CORNERS[:3]:
        svc.add_corner(x, y)
    with pytest.raises(ValueError):
        svc.build()
    svc.undo()
    assert svc.next_corner == "bottom_right"
    svc.add_corner(*DEFAULT_CORNERS[2])
    svc.add_corner(*DEFAULT_CORNERS[3])
    assert svc.complete
    with pytest.raises(ValueError):
        svc.add_corner(0.5, 0.5)
    cal = svc.build(camera_id="1", frame_size=(1280, 720))
    assert cal.transformer().to_surface(DEFAULT_CORNERS[0])[0] == pytest.approx((0, 0), abs=1e-9)


def test_calibration_persistence(tmp_path):
    mgr = SettingsManager(tmp_path / "settings.json")
    assert mgr.load().calibration is None
    s = AppSettings(calibration=DeskCalibration(image_corners=DEFAULT_CORNERS))
    s.detection.min_travel = 0.15
    s.layout.width = 0.8
    mgr.save(s)
    loaded = mgr.load()
    assert loaded.calibration.image_corners == [tuple(c) for c in DEFAULT_CORNERS]
    assert loaded.detection.min_travel == 0.15 and loaded.layout.width == 0.8


def test_corrupt_settings_fall_back_to_defaults(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text('{"calibration": {"image_corners": [[0,0]]}}')
    assert SettingsManager(path).load() == AppSettings()
    assert (tmp_path / "settings.corrupt.json").exists()
