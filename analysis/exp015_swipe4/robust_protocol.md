# Four-gesture robust baseline protocol

## Objective

Recognize the user's four intended DOWN, LEFT, RIGHT, and TOP finger actions while treating a relaxed-finger hand tilt as REST/no command. Natural wrist/hand tilt is allowed when it accompanies the intended finger action. Determine whether EMG carries the command information, whether the LIS3DH alone can create the observed false RIGHT response, and whether combining modalities helps across re-wear sessions.

## Protocol outline

- Four independently recorded sessions. Remove and re-wear the band between sessions; record session date, orientation, and any placement adjustment.
- Sessions 1–3 are development data. Compare candidates with whole sessions kept together. Session 4 is untouched until the chosen pipeline is frozen.
- Per session: 20 randomized examples of each intended gesture, plus 10 relaxed-finger tilt-only controls in each direction. A tilt-only control is labeled REST and stores its direction separately.
- Each action cue lasts 1.5 seconds and is preceded by 2.5 seconds of REST. Insert an 8-second break after every 30 actions. Capture one continuous BLE stream and preserve raw notification bytes, processed CSV, and cue sidecar.
- Compare EMG-only, accelerometer-only, and fusion inputs. Start with the existing shrinkage LDA/RBF SVM and add classic temporal features before trying sequence models.
- Report per-gesture event precision/recall, event-level macro-F1, tilt-only false commands, false commands during ordinary activity, and detection latency. Do not use overlapping windows as independent trials.

The repetition count and pass criteria are project choices, not universal standards. A planned minimum engineering target is at least 90% event accuracy on the untouched re-wear session, with no tilt-only control producing a command. A longer negative-activity run is needed to estimate accidental command rate.

## Session 1 cue meanings

- `DOWN`, `LEFT`, `RIGHT`, `TOP`: perform the specific intended finger action used for that label in the supplied browser recording. Natural tilt may accompany it; the finger action defines the class.
- `TILT` plus an arrow: keep fingers relaxed and tilt the hand only in the arrow direction. This is a REST/no-command negative control.
- `REST`: relax fully; do not make a target gesture.

The cue screen displays one large cue word at a time. The tilt control uses a direction arrow so the displayed word remains `TILT`.

## Session 1 artifacts and disposition

The generic next-session configuration is `robust_session_template.json`. Copy it and assign a unique acquisition ID before each recording. Recording paths and participant-specific notes stay in local ignored files.

The first launch was stopped after it had recorded a short partial sample. The user asked to discard that take. It is marked excluded and must not be segmented, fitted, or evaluated. Preserve the original filtered CSV, raw BLE packet log, and cue sidecar for audit; the partial capture is not experiment data. The corrected Session 1 capture is complete and saved under `supplier_toolkit/adapted/training_data/`.

Session 1 QC confirms 512,620 samples, continuous EMG/IMU counters, configured-rate host arrival near 1,000 Hz, complete cues, and 20 intended gesture trials per class plus 10 tilt-only negative controls per direction. The user confirmed following the full sequence. Keep this entire wearing as one development group. Candidate models have since been evaluated with group-held-out repetitions within the session; none is locked for deployment.

## Session 1 accuracy checks

The previously locked LDA+fusion model, evaluated without fitting on Session 1, classified 64/120 action intervals correctly (53.3% event accuracy): 27/80 intended gestures and 37/40 tilt-only controls. This shows poor transfer from the older recording to the new session's intended finger-action labels.

Within-session five-fold stratified group CV held out each complete action repetition, keeping its overlapping windows together. Best candidate was SVM+fusion at 90/120 event accuracy (75.0%), with 62/80 gesture events correct (77.5%) and 12/40 tilt-only controls falsely called commands. EMG-only SVM scored 81/120 (67.5%), 44/80 gestures (55.0%), and 3/40 tilt controls falsely called. Accelerometer-only SVM scored 44/120 (36.7%) and called 32/40 tilt controls commands. LEFT recall for the best within-session candidate was 9/20 (45%).

The within-session score is optimistic for real deployment because all folds share one wearing. Session 1 has been used for model/feature screening; reserve Session 4 for a final locked evaluation. The frozen-model and grouped-CV reports are `session1_model_comparison.json` and `session1_within_session_grouped_cv.json`. No controller was trained for live action and no HID events were sent.

No firmware changes are part of this protocol.
