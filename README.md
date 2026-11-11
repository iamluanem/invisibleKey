# Invisible AI Keyboard

Turn a plain desk into a keyboard using only a webcam: hand tracking, desk calibration
(homography), temporal tap detection and OS keyboard injection — 100% local.

> **Status: Phase 2 of 5 — live camera + MediaPipe + calibration in a local debug app.**
> API and React dashboard come next (see Roadmap).

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
| CameraManager | `invisiblekey/vision/camera.py` (threaded, newest-frame) |
| HandTrackingService | `invisiblekey/vision/hand_tracking.py` (MediaPipe Tasks, VIDEO mode) |
| Recording / replay | `invisiblekey/recording.py` (JSONL landmarks) |
| Emergency hotkey | `invisiblekey/safety.py` |
| Debug app | `invisiblekey/app.py` + `vision/overlay.py` |

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
pip install -e ".[dev,vision,input]"
pytest -q
python -m invisiblekey.simulation --text 'Hello, World!' --noise 0.002   # no camera needed
python -m invisiblekey.vision.model      # one-time model download (~8 MB), then offline
python -m invisiblekey.app               # live app
```

On Intel Macs pip installs MediaPipe 0.10.21, the last release with Intel Mac wheels. It also
pins NumPy below 2. Apple Silicon, Windows and Linux get MediaPipe 1.x. Both expose the same Tasks API.

### Debug app

1. **Calibrate:** click the 4 corners of the desk typing area (top-left, top-right, bottom-right,
   bottom-left as seen on screen), then press ENTER. The calibration is saved in
   `~/Library/Application Support/InvisibleKey/settings.json`.
2. **Type:** fingertip dots change color by phase (grey hover · blue moving · orange descending
   · red contact · green lifting). Green key flash = typed, yellow = uncertain (not typed).
3. **Keys** (window focused, OS output OFF): `C` recalibrate · `O` toggle OS output ·
   `+`/`-` sensitivity · `[`/`]` keyboard size · `I J K L` move keyboard · `S` save ·
   `Z` clear · `R` record landmarks · `Q` quit.
4. While OS output is ON, the window ignores keys (they could be our own injected keys) except
   `ESC`. **Global emergency stop from any app: Control + Option + K.** Output refuses to turn on
   if that hotkey cannot be registered.

`python -m invisiblekey.app --list-cameras` lists cameras, `--camera N` picks one, and `--no-mirror` is
for a camera that looks down from behind the hands.

### macOS permissions (System Settings → Privacy & Security)

| Permission | Needed for | Grant to |
|---|---|---|
| Camera | video capture | your terminal (Terminal / iTerm / Warp / VS Code) |
| Accessibility | typing into other apps | same terminal |
| Input Monitoring | global emergency hotkey | same terminal |

### Record real sessions, tune offline

Press `R` in the app to save landmarks to `recordings/*.jsonl` along with the calibration in use. Then
replay them deterministically through the pipeline, for example after changing thresholds:

```bash
python -m invisiblekey.recording recordings/session-123.jsonl --expected 'hello world'
```

This is how real-world accuracy gets measured. Synthetic numbers only validate logic.

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
2. ✅ Camera + MediaPipe Hand Landmarker, calibration from live frames, recording/replay, debug app,
   global emergency hotkey
3. FastAPI + WebSocket server, React/TS/Tailwind dashboard (preview, overlay, wizard, stats)
4. macOS Accessibility check from the UI, Windows/Linux input backends
5. Language-model decoding (probabilistic key correction), ONNX temporal tap classifier
