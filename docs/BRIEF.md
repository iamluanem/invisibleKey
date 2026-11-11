# Project: Invisible AI Keyboard — Camera-Based Touchless Physical Keyboard Replacement

Act as a Principal AI Engineer, Computer Vision Specialist, and Senior Full-Stack Software Architect.

I want you to design and implement a functional MVP of **Invisible AI Keyboard**, an AI-powered application that transforms an ordinary desk surface into a virtual keyboard using only a laptop camera or external webcam.

## 1. Product Vision

The goal is to eliminate the need for a physical keyboard.

Users should be able to position their hands on a normal desk, simulate typing on an invisible QWERTY keyboard, and have their finger movements translated into actual keyboard input on their computer.

The system must work without physical keyboards, special gloves, depth sensors, projectors, or additional hardware beyond a standard camera.

The application should be designed as a real product, not merely a hand-tracking demonstration.

## 2. Core Functional Requirements

### A. Real-Time Hand Tracking

- Capture video using OpenCV.
- Detect up to two hands simultaneously using MediaPipe Hand Landmarker.
- Track 21 landmarks per hand.
- Identify fingertips and their positions.
- Support hand movement across a calibrated typing surface.
- Display landmarks and tracking confidence in a debug interface.
- Process video locally to preserve privacy.

### B. Desk Surface Calibration

Create a calibration workflow that allows users to:

1. Activate the camera.
2. Define the physical desk typing area.
3. Set four reference corners.
4. Map the selected surface to a normalized coordinate system.
5. Configure a standard QWERTY keyboard layout.
6. Adjust the layout size and position.
7. Save calibration settings locally.

Use perspective transformation or homography where appropriate.

Support recalibration when the camera moves.

### C. Virtual Keyboard Engine

Implement a configurable keyboard mapping engine supporting:

- QWERTY alphabetic keys.
- Numbers.
- Space.
- Enter.
- Backspace.
- Shift.
- Basic punctuation.
- Repeated key presses.
- Simultaneous tracking of multiple fingers.

Each key must have a defined region in the calibrated coordinate space.

### D. AI-Based Tap Detection

This is the most important technical component.

Implement a temporal gesture recognition pipeline capable of distinguishing:

- Finger hovering.
- Finger moving between keys.
- Finger descending toward the desk.
- Intentional tap gestures.
- Finger lifting after a tap.

Use landmark trajectories, motion velocity, acceleration, temporal windows, and confidence thresholds.

Start with a rule-based detection engine.

Design an interchangeable ML-based detector interface for future improvements using a trained temporal classification model.

Do not assume that 2D landmarks alone can reliably prove physical contact with the desk.

Provide confidence scores, configurable thresholds, and clear handling of uncertain events.

### E. Keyboard Input Integration

When a valid tap is detected:

1. Identify the corresponding virtual key.
2. Generate a keyboard event.
3. Send the event to the active application using a supported operating-system input API.
4. Prevent accidental duplicate events.
5. Allow intentional repeated typing.

Include an explicit enable/disable control and an emergency stop shortcut.

Support Windows initially, with a modular architecture for macOS and Linux.

### F. User Interface

Build a modern dashboard using React and TypeScript.

The interface should include:

- Live camera preview.
- Hand landmark visualization.
- Virtual keyboard overlay.
- Calibration wizard.
- Detected key visualization.
- Real-time text testing area.
- Camera selection.
- Detection sensitivity settings.
- Typing performance statistics.
- Start/Stop controls.

The design should be minimal, futuristic, and suitable for a professional AI product demo.

## 3. Technology Stack

Backend:
- Python 3.11+
- OpenCV
- MediaPipe
- NumPy
- FastAPI
- WebSockets
- pynput or platform-specific input APIs
- Pydantic
- pytest

Frontend:
- React
- TypeScript
- Vite
- Tailwind CSS

Machine Learning:
- Rule-based temporal gesture detection for MVP.
- ONNX Runtime as a future optional inference engine.
- Pluggable model interface for trained tap classifiers.

## 4. Proposed Architecture

Create separate modules for:

- CameraManager
- HandTrackingService
- DeskCalibrationService
- CoordinateTransformer
- VirtualKeyboardLayout
- GestureDetectionEngine
- TapConfidenceEstimator
- KeyboardEventDispatcher
- SettingsManager
- WebSocketEventServer
- PerformanceMonitor

Ensure low coupling, clear interfaces, and testability.

## 5. Performance Requirements

Target:

- 30 FPS camera processing on suitable consumer hardware.
- Low-latency visual feedback.
- Minimal CPU overhead.
- No cloud dependency for core typing.
- Robust handling of temporary hand-tracking loss.

Measure end-to-end input latency and false-positive rates rather than claiming guaranteed performance.

## 6. Testing

Implement automated tests for:

- Coordinate mapping.
- Keyboard region detection.
- Tap state transitions.
- Duplicate key prevention.
- Repeated key presses.
- Calibration persistence.
- Missing landmark handling.
- Keyboard event dispatching.

Include a simulation mode using recorded or synthetic landmark sequences so core behavior can be tested without a camera.

## 7. MVP Acceptance Criteria

The MVP is complete when a user can:

1. Launch the application locally.
2. Activate a webcam.
3. Calibrate a desk surface.
4. See their hands tracked in real time.
5. Perform deliberate tap gestures over mapped keys.
6. See detected letters appear in a text testing field.
7. Enable keyboard input for other applications.
8. Disable input immediately when needed.

Clearly document limitations regarding camera angle, lighting, occlusion, and physical contact detection.

## 8. Deliverables

Generate:

1. Complete project architecture.
2. Backend and frontend source code.
3. Directory structure.
4. Dependency configuration.
5. Local installation instructions.
6. Fully implemented calibration workflow.
7. Hand tracking and tap detection.
8. Keyboard event integration.
9. Automated tests.
10. README with architecture and limitations.

Do not generate pseudocode or empty placeholder implementations.

Implement the project incrementally, beginning with a runnable end-to-end MVP.

Prioritize actual functionality, testability, and accurate technical behavior over visual effects.

Start by explaining the architecture and then generate the complete implementation.