# Invisible AI Keyboard

Turn a plain desk into a keyboard using only a webcam: hand tracking, desk calibration
(homography), temporal tap detection and OS keyboard injection — 100% local.

> **Status: Phase 1 of 5 — core engine + simulation.** Runs and is tested without a camera.
> Camera/MediaPipe, API and React dashboard come next (see Roadmap).

## Architecture

```
Camera ─► HandTracking (MediaPipe, phase 2) ─► FrameObservation (21 landmarks/hand)
                                                   │
                         CoordinateTransformer ◄───┤ (desk homography from 4 corners)
                                                   ▼
                     per-finger FingerSample: press signal + fingertip/knuckle on surface
                                                   ▼
             TapDetector (Protocol) ── RuleBasedTapDetector now / ONNX classifier later
                                                   ▼  TapCandidate
     VirtualKeyboardLayout.hit_test ─► TapConfidenceEstimator ─► accepted / uncertain / rejected
                                                   ▼
                   KeyPressPolicy (duplicates vs. deliberate repeats)
                                                   ▼
     KeyboardEventDispatcher ─► TextBuffer (always)  +  OS backend (pynput, only when enabled)
                                                   ▼
                                  PerformanceMonitor (measured, not claimed)
```

| Module | File |
|---|---|
| CoordinateTransformer | `invisiblekey/geometry.py` |
| DeskCalibrationService | `invisiblekey/calibration.py` |
| VirtualKeyboardLayout | `invisiblekey/layout.py` |
| GestureDetectionEngine | `invisiblekey/gestures/` (`base.py` interface, `rule_based.py`) |
| TapConfidenceEstimator | `invisiblekey/confidence.py` |
| Duplicate policy | `invisiblekey/policy.py` |
| KeyboardEventDispatcher | `invisiblekey/dispatcher.py` |
| SettingsManager | `invisiblekey/settings.py` |
| PerformanceMonitor | `invisiblekey/performance.py` |
| Pipeline | `invisiblekey/pipeline.py` |
| Simulation | `invisiblekey/simulation.py` |

### How a tap is detected

A single 2D camera **cannot prove a finger touched the desk**. The engine detects the
*kinematic signature* of a tap on a **press signal**: how far the fingertip sits below its own
reference joint, normalized by hand size. Whole-hand motion moves all joints together, so the
press signal stays flat.

```
HOVER/MOVING ─fast descent─► DESCENDING ─abrupt stop─► CONTACT (tap) ─lift─► LIFTING ─rearm─► HOVER
```

- **One Euro filter**: heavy smoothing at rest, low lag during fast motion.
- **Adaptive threshold**: each finger estimates its own landmark jitter while hovering. A tap
  must travel `max(min_travel, k·σ)`. When tracking gets noisy, the system *misses* taps
  instead of *inventing* them.
- **Re-arm**: a finger must lift before it can tap again. That kills bounces and still allows "ll".
- **Confidence** = weighted geometric mean of velocity, deceleration, travel, lateral stability,
  tracking score and key centrality. `uncertain` taps show in the UI but are never typed.
- **Shift** is one-shot (tap Shift, then the letter), because holding a chord is unreliable.

## Measured results (synthetic data)

Reproduce: `python -m invisiblekey.simulation --text "..." --noise 0.003`.
Sentence: "The quick brown fox jumps over the lazy dog, 1234." ×5 seeds, plus 100 s of idle hands.

| Landmark jitter (σ, image units) | Char error rate | Spurious keys while typing | False keys / 100 s idle | Detection delay p50 |
|---|---|---|---|---|
| 0.000 | 0 % | 0 | 0 | 33 ms |
| 0.002 | 0.4 % | 0 | 0 | 40 ms |
| 0.003 | 9 % (missed taps) | 0 | 0 | 43 ms |
| 0.004 | 35 % | 2 | 7 | 53 ms |

These are **synthetic** numbers. They validate the logic, not real-world accuracy. Real numbers
depend on camera angle, lighting and the user, and will be measured in phase 2.

## Install & run (macOS)

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
python -m invisiblekey.simulation --text "Hello, World!" --noise 0.002
```

## Safety

- OS output is **disabled by default**. The text test area works without it.
- `emergency_stop()` disables output and clears pending Shift. Any OS injection error
  disables output automatically.
- macOS: injecting keys needs **System Settings → Privacy & Security → Accessibility**
  permission for the terminal/app running the backend.

## Known limitations

- **Contact is inferred, never measured.** Hovering a finger, then stopping it abruptly, can look like a tap.
- **Camera angle**: a laptop webcam sees the desk at a grazing angle. An external webcam looking
  down at ~45–90° works much better.
- **Frame rate**: at 30 FPS a fast tap lasts 2–4 frames, so expect deliberate hunt-and-peck speeds,
  not touch-typing speeds.
- **Occlusion**: fingers hidden behind the hand lose tracking, and the tracker resets them safely.
- **Lighting**: low light raises landmark jitter, which raises the adaptive threshold (more misses).
- No key auto-repeat on hold. Shift chords are replaced by one-shot Shift.

## Roadmap

1. ✅ Core engine, rule-based tap detection, simulation, tests
2. Camera + MediaPipe Hand Landmarker, calibration from live frames, landmark recording/replay
3. FastAPI + WebSocket server, React/TS/Tailwind dashboard (preview, overlay, wizard, stats)
4. macOS input integration (Accessibility check, global emergency hotkey), Windows/Linux backends
5. Language-model decoding (probabilistic key correction), ONNX temporal tap classifier
