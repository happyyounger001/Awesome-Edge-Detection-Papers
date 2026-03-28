# Fencing Lunge AI Trainer — Software Development Specification

**Version:** 1.0  
**Language:** English  
**Audience:** Software engineers, AI engineers, QA engineers, product designers, technical writers, and autonomous coding systems.

---

## 1. Purpose and Scope

This specification describes, in implementation-level detail, a complete software system for analyzing fencing lunge (弓步) training videos, generating coaching feedback, learning quality standards from curated samples, and presenting results through a desktop GUI.

The goal of this document is to enable **any programmer or AI system** to re-implement a functionally equivalent product, including:

- Product requirements and user workflows.
- System architecture and module boundaries.
- Data contracts and feature schemas.
- Algorithm logic (state machines, scoring, learning, reporting).
- UI layout, behavior, interaction states, and style semantics.
- Error handling, validation, and testing strategy.

This document reflects the currently implemented system and its intended evolution path.

---

## 2. Product Vision

### 2.1 Core Product Objective

Build a practical AI-assisted fencing training application that can:

1. Detect and count complete lunge actions from video.
2. Evaluate lunge quality with a learned standard (not static hard-coded pass/fail only).
3. Provide actionable coaching feedback (single highest-priority issue + recommendation).
4. Let users build/improve standards through a **human-in-the-loop learning workflow**.

### 2.2 Product Principles

- **Coach-first UX:** Clear, immediate, actionable feedback.
- **Human-controlled learning:** Samples must be manually confirmed before entering the learning set.
- **Stage-aware movement understanding:** Lunge is a sequence, not a single-frame event.
- **Cross-role usability:** Works for both children and adults via normalized features.
- **Traceable outputs:** Every analysis should produce reproducible artifacts (overlay video, report, metrics, JSON).

---

## 3. User Roles and Main Use Cases

### 3.1 Roles

- **Coach / Parent:** Uses the training assistant to evaluate practice videos and review feedback.
- **Analyst / Curator:** Uses learning mode to inspect candidate “good sample” videos and decide inclusion.
- **Developer / Integrator:** Uses CLI and JSON outputs for automation and system extension.

### 3.2 Primary Use Cases

1. Import a training video and run analysis.
2. Watch overlay video with skeleton and warnings.
3. Read realtime and report-level coaching feedback.
4. Open learning system, analyze candidate samples, include/reject each sample.
5. Rebuild quality standard from included samples.
6. Apply learned standard to training assistant in one click.

---

## 4. High-Level Architecture

The system is organized into three layers.

## Layer A — Pose & Features
- `pose/estimator.py`: MediaPipe extraction + smoothing.
- `pose/models.py`: Pose frame/sequence structures.
- `feature_extractor.py`: Video-level normalized feature extraction.

## Layer B — Learning & Scoring
- `learning_system.py`: Sample aggregation and standard generation pipeline.
- `quality_standard.py`: Distribution summarization and standard serialization.
- `quality_scorer.py`: Scoring and top-issue feedback against learned distributions.

## Layer C — Training Assistant & UX
- `rules/evaluator.py`: Sequence evaluation + lunge state machine + per-frame assessment.
- `analysis/engine.py`: Orchestration of extraction → evaluation → scoring → overlay → report.
- `ui/`: Desktop GUI windows and interaction logic.
- `report/generator.py`: Human-readable HTML report + charts + CSV.

---

## 5. Repository / Module Map

- `main.py` — GUI entry point.
- `training_assistant.py` — CLI training analysis runner.
- `learning_system.py` — CLI + reusable learning builders.
- `analysis/`
  - `models.py` — `TimingResult`, `LungeQuality`, `FrameAssessment`, `AnalysisResult`.
  - `engine.py` — central pipeline coordinator.
- `rules/evaluator.py` — sequence-level biomechanics and state machine logic.
- `pose/` — keypoint extraction and structured pose sequence storage.
- `feature_extractor.py` — normalized kinematic features for scoring/learning.
- `quality_standard.py` — statistical profile generation and config application.
- `quality_scorer.py` — score groups + top issue/advice derivation.
- `report/generator.py` — report generation and artifact writing.
- `ui/main_window.py` — training assistant GUI.
- `ui/learning_window.py` — sample-analysis learning GUI.
- `ui/video_overlay.py` — overlay rendering for training and learning modes.
- `ui/progress_dialog.py`, `ui/report_window.py` — supporting dialogs/windows.
- `tests/` — regression tests and synthetic sequence fixtures.

---

## 6. Runtime Modes

The product has two operational modes with intentionally different semantics.

## 6.1 Training Assistant Mode (`training_assistant_mode`)
Purpose: Daily training feedback.

- Displays warning semantics (e.g., head tilt warning, order warning).
- Produces score state and coaching recommendation.
- Counts complete lunges for session tracking.

## 6.2 Learning Sample Mode (`learning_sample_mode`)
Purpose: Curate and learn from candidate standard samples.

- **No judgmental training-warning language** in the sample review UI.
- Focuses on stage, feature observation, and sample inclusion decision.
- Sample is excluded by default until explicitly included.

---

## 7. End-to-End Workflows

## 7.1 Training Analysis Workflow
1. User imports a video.
2. Player enters `LOADING` state and blocks playback.
3. Pose sequence is extracted.
4. Player enters `READY`.
5. User plays/seeks/replays video.
6. For each displayed frame, overlay + realtime panel are updated.
7. On analysis command, engine computes full outputs:
   - evaluation metrics
   - quality scoring
   - coaching feedback
   - overlay export
   - report export

## 7.2 Learning Workflow (Human-in-the-Loop)
1. User imports candidate sample video(s), labels category.
2. Select a sample and click **Analyze Sample**.
3. System extracts pose, stage timeline, features, and learning overlay artifacts.
4. User reviews:
   - sample playback
   - frame-level features
   - lunge-level summary
5. User chooses:
   - **Include in Learning Set** → merge + rebuild standard.
   - **Reject Sample** → never enters learning aggregate.
6. User can apply the resulting standard to training assistant.

---

## 8. Data Contracts

## 8.1 Frame Assessment Contract
Each frame should have:
- frame index, timestamp
- current lunge index
- completed lunge count
- state machine stage
- current status text/icon
- top issue and coaching advice
- detail flags (thigh warning, order warning, head tilt warning)

## 8.2 Lunge Quality Contract
Must include:
- posture/timing/stability/coordination/overall scores
- overall status
- top issue
- top advice
- deviations list

## 8.3 Learning Structured Sample Contract
Each curated sample should be serializable as:

```json
{
  "lunge_index": 1,
  "category": "training_standard",
  "static_baseline": {},
  "dynamic_series": {},
  "extended_phase": {},
  "pre_return_phase": {},
  "summary": {}
}
```

### 8.3.1 `static_baseline`
Pre-movement baseline posture references (angles, stance width, neutral alignments).

### 8.3.2 `dynamic_series`
Time-series sampled at a fixed interval (e.g., 0.05s) with stage labels and key normalized features.

### 8.3.3 `extended_phase`
At or near max extension: front knee angle, trunk lean, stride norm, wrist extension norm, peak speed hints.

### 8.3.4 `pre_return_phase`
Stability immediately before recovery: head/trunk variability, wrist stability, support quality proxies.

### 8.3.5 `summary`
Session-level lunge summary, including high-priority biomechanical dimensions for learning.

---

## 9. Lunge Counting Logic (Canonical Definition)

## 9.1 Business Rule
A complete lunge is:

**Ready/stand → leg out + arm extend → active lunge → return → back to ready = 1 count**

## 9.2 Non-Count Cases
Do **not** count:
- micro-adjustments
- small local jitter
- backward motions
- leg-only movement
- arm-only movement
- incomplete return-to-ready

## 9.3 State Machine
Recommended strict states:
- `IDLE`
- `PREPARE`
- `LEG_OUT`
- `ARM_EXTEND`
- `LUNGE_ACTIVE`
- `HOLD`
- `RETURN`
- back to `IDLE` with count increment

## 9.4 Gate Conditions
- Minimum movement amplitude thresholds.
- Minimum confirmation frame counts.
- Forward-direction gate (reject backward-dominant motion).
- Return-to-ready gate before increment.

## 9.5 Stability Against False Positives
Use combined temporal + magnitude gates, not a single threshold.

---

## 10. Feature Extraction Specification

All major features should be **normalized**, minimizing dependence on height, camera distance, and pixel resolution.

## 10.1 Core Normalized Features
- Front/rear knee angles
- Trunk lean angle
- Head tilt angle
- Normalized stride length
- Normalized wrist-forward distance
- Hand start time / foot start time / delta
- Stability standard deviations (wrist/elbow/shoulder/head/trunk)
- Stable hold duration

## 10.2 Priority Biomechanical Learning Dimensions
Treat these as **priority learning dimensions**, not hard-coded forced constants:
- Trunk angle (beta-like)
- Front knee angle (alpha-like)
- Extension line quality
- Time-to-extension (t-like)
- Speed proxy (v-like)
- Terminal/control deviation hint (delta-like)
- Pre-return support stability

---

## 11. Scoring and Feedback Logic

## 11.1 Score Groups
- posture
- timing
- stability
- coordination
- overall (weighted aggregate)

## 11.2 Distribution-Based Scoring
Current value is scored against learned distribution bands (`p10/p25/p75/p90`).

## 11.3 Top-Issue Policy
From prioritized feature deviations, select one highest-priority issue and emit one concise recommendation.

Outputs include:
- status text (`good`/`needs improvement`)
- `top_issue`
- `top_advice`
- deviations list

---

## 12. Video Overlay and Rendering

## 12.1 Training Overlay
- Draw skeleton aligned to current frame dimensions.
- Subtitle content limited to:
  - lunge number
  - active warning lines only
- Text anchor fixed in UI overlay coordinates (top-left stack).

## 12.2 Learning Overlay
- Draw skeleton and stage/frame labels.
- Avoid training-warning semantics.

## 12.3 Orientation Handling
Player should normalize orientation via metadata if available.

---

## 13. Player State Machine and Interaction Rules

Canonical states:
- `IDLE`
- `LOADING`
- `READY`
- `PLAYING`
- `PAUSED`
- `ENDED`

Rules:
- Never transition `LOADING -> PLAYING` directly.
- Playback buttons are disabled or no-op while loading.
- Seek is blocked during loading.
- Replay from end returns to frame 0.

---

## 14. GUI Specification

## 14.1 Training Window
### Layout
- Left: video + playback controls.
- Right top: realtime analysis.
- Right middle: current rules/standard panel.
- Right lower: controls and parameters.

### Required realtime fields
- Current lunge number
- Completed count
- Current stage
- Current status (green/red)
- Highlighted top issue card
- Detail items (thigh, stability, order, head)

## 14.2 Learning Window
### Layout
- Sample queue + category selection.
- Sample analysis player.
- Realtime stage/feature panels.
- Include/reject controls.
- Standard summary and metrics table.

### Required behavior
- Sample is not included by default.
- Inclusion triggers aggregate merge + standard rebuild.

## 14.3 Unified Visual Semantics
- Blue: primary action/system info
- Green: success/qualified
- Red: problem/warning
- Orange: pending/attention

---

## 15. Reporting Specification

Generated artifacts per analyzed video:
- `report_zh.html`
- charts (timing/stability)
- `metrics.csv`
- overlay video
- keyframe snapshots

Report sections should include:
- session summary
- timing breakdown
- quality table
- charts
- deviations
- top issue and recommendation

---

## 16. CLI Interfaces

## 16.1 Training Assistant
```bash
python -m training_assistant --video input.mp4 --standard outputs/quality_standard.json
```

## 16.2 Learning Builder
```bash
python -m learning_system --input data/learning_samples --output outputs/quality_standard.json
```

---

## 17. Error Handling Requirements

- Missing/invalid video: show clear error dialog.
- Pose detection failure in some frames: mark invalid frame; continue if possible.
- Empty sample set on learning rebuild: block rebuild with explicit message.
- Prevent standard import if no learned standard is available.

---

## 18. Testing Strategy

## 18.1 Unit Tests
Minimum test domains:
- config loading
- learning standard generation
- scoring/feedback outputs
- state machine lunge counts
- report generation artifacts
- build command composition

## 18.2 Scenario Tests
- Seek/pause/replay consistency.
- Loading-state playback lock.
- Skeleton alignment after resize/seek.
- Learning sample inclusion/rejection integrity.

## 18.3 Regression Focus
- False-positive lunge count reduction.
- Overlay text placement stability.
- Orientation behavior consistency.

---

## 19. Non-Functional Requirements

- Desktop responsiveness during playback.
- Bounded UI updates during playback to avoid rendering overload.
- Reproducible output files for analysis auditability.
- Maintainable module boundaries and serializable schemas.

---

## 20. Implementation Notes for Rebuilders

If re-implementing from scratch:

1. Build pose extraction and sequence model first.
2. Implement strict lunge state machine with dual-signal gating (leg + arm).
3. Add normalized feature extraction and distribution-based scoring.
4. Add report and overlay export pipeline.
5. Build training UI with clear player state machine constraints.
6. Build learning UI with mandatory human-confirmed inclusion.
7. Add test fixtures and regression tests before packaging.

---

## 21. Deliverables Checklist (Definition of Done)

- [ ] Training mode: stable playback and accurate counting.
- [ ] Learning mode: review-first, include/reject gating.
- [ ] Structured sample JSON with 4-stage segmentation.
- [ ] Quality standard JSON rebuild from curated samples.
- [ ] Realtime and report-level top issue/advice.
- [ ] Overlay and report artifacts generated successfully.
- [ ] CLI flows operational.
- [ ] Test suite and compile checks pass in prepared environment.

---

## 22. Future Extensions (Recommended)

- Multi-lunge segmentation per sample clip.
- Better direction inference via camera-aware calibration.
- Weapon-tip tracking and endpoint accuracy metrics.
- Model-assisted phase classification (hybrid rules + ML).
- Cross-session athlete progression analytics.

---

## 23. Glossary

- **Lunge (弓步):** Fencing attack movement involving coordinated leg extension and arm extension.
- **Overlay:** Annotated video frame with skeleton and textual markers.
- **Standard:** Learned statistical profile of preferred movement features.
- **Top issue:** Highest-priority deviation selected for coaching.
- **Learning sample mode:** Non-judgmental curation workflow for standard learning.

---

_End of specification._
