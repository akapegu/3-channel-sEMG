# sEMG hand control: literature review and prototype implications

Prepared for Abinash Pegu · 1 November 2026

## 1. What the literature says about this project

**Gesture classification from surface EMG is an established field. Reliable, comfortable, user-programmable control during everyday use remains a harder engineering problem.** Your three-channel forearm band is a reasonable instrument for investigating that problem. It is not evidence that you can already reconstruct all finger joints, transfer a wrist model directly, or offer arbitrary gestures with two examples.

The useful question is no longer “can a neural network classify EMG?” It is: **which intentional commands can this particular band recognise reliably after removal and re-wearing, with little calibration, without firing during ordinary activity?**

Your advantage at this stage is access to real measurements and the ability to build and evaluate the complete interaction. Public datasets accelerate learning; they do not replace characterisation of your acquisition hardware, placement, users, and intended environment. A convincing result could be modest in class count yet strong in repeatability and false-command performance.

### Scope and evidence standard

This is a broad, project-focused literature review, not a registered systematic review or an exhaustive catalogue of every sEMG paper. Searches started from EgoEMG and emg2pose, followed relevant references backward, and searched later work on adaptation, pose estimation, channel selection, and foundation models. Sources were screened through 1 November 2026.

The bibliography distinguishes **F** (full text or substantive methods/results inspected), **A** (abstract or institutional publication record inspected), and **P** (official project, repository, or dataset documentation inspected). F does not mean independently reproduced. No model was trained and no large dataset was downloaded for this review. Publication status and versions matter: preprints, workshop papers, supplier claims, and peer-reviewed experiments are not interchangeable.

Citation numbers below refer to the linked bibliography. Related-work connections are explanatory groupings unless explicitly described as a citation relationship. No complete forward-citation count is claimed.

## 2. Six different problems that often get mixed together

| Problem | Model output | What success establishes | What it does not establish |
|---|---|---|---|
| Fixed-vocabulary classification | Fist, pinch, open, rest | Discrimination among the tested classes and conditions | Arbitrary user-defined gestures |
| New-user adaptation | Existing classes for a new wearer | Reduced calibration for known actions | Recognition of a never-trained class |
| Custom-gesture enrollment | A new class from a few demonstrations | New-class learning under the tested protocol | Every possible hand action is distinguishable |
| Continuous pose regression | Joint angles or fingertip coordinates over time | Kinematic estimation against a reference system | Exact force, intent, contact, or reliable commands |
| Sequence recognition | Text, strokes, temporal command sequences | Decoding with temporal/task constraints | General unconstrained hand tracking |
| Robot or musculoskeletal control | Simulated tendons, robot actions, policy inputs | The evaluated control or retargeting task | Improved physical robot learning without an actual comparison |

Report these tasks separately. Classification accuracy, angular error, character error rate, target-acquisition rate, and robot task success cannot form one leaderboard.

## 3. The two papers you sent

### emg2pose: the most useful large-scale pose benchmark [1]

Salter et al. provide 370 hours from 193 users, 16 bipolar wrist channels at 2 kHz, and motion-capture-derived hand labels. Their temporal model combines convolutional features with autoregressive recurrent prediction. Crucially, **regression** starts without the true initial pose; **tracking** is given it. Reported held-out-user angular MAE is approximately 12.2° for regression versus 7.7° for tracking; held-out users plus movement stages are harder. Labels originate from 60 Hz capture and are interpolated, rather than independently measured at 2 kHz. Invalid pose labels are masked. Hours count hands separately. Most users re-wore the device during collection. Use this benchmark to study evaluation, temporal modelling, and pose supervision—not as proof that a three-channel forearm model inherits its performance. Code: [facebookresearch/emg2pose](https://github.com/facebookresearch/emg2pose).

### EgoEMG: multimodal, continuous pose, and version-sensitive [2]

Xi et al. pair wrist EMG with egocentric imagery and hand-pose references. The paper describes 41 participants, more than 10 hours, 60 action categories, eight EMG channels per wrist at 2 kHz, and a transformer-based pose estimator. Its target includes continuous finger and wrist angles; “60 gestures” is not automatically a 60-class recognition result. The current repository advertises a newer 53-person, 18-plus-hour release, so dataset counts must be tied to a version. Vision-only, EMG-only, and fusion results are useful ablations, but their errors cannot be compared directly with emg2pose under different splits and metrics. Start with a small data example and loader inspection. Training instructions include multi-GPU configurations; deployment on an ESP32 is not demonstrated. Repository: [zhenqis123/EgoEMG](https://github.com/zhenqis123/EgoEMG).

## 4. Citation trails worth following

### Established decoding and evaluation

Hudgins [3] and Englehart–Hudgins [4] → engineered features and real-time pattern recognition → NinaPro [5–7] and CNN/transfer-learning work [8–11] → cross-user benchmarks such as EMGBench [12]. This line is directly useful for your first reproducible baseline.

### Continuous hand reconstruction

NeuroPose [13] and Sensing the Full Dynamics [14] are among the earlier methods discussed by emg2pose [1]. Later work explores discrete pose representations (VQ-MyoPose [15]), multimodal estimation (EgoEMG [2]), musculoskeletal retargeting (emg2tendon [16]), and dexterous teleoperation (DexEMG [17]). These papers solve related but different targets with substantially different sensing systems.

### Population models, personalization, and new gestures

Transfer learning [11], few-shot recognition [18], confidence/metric learning [19–20], and cross-user temporal decoding [21] address different parts of enrollment and robustness. CPEP [22] explicitly builds on emg2pose using pose–EMG alignment. EMG-Adapt [23] and electrode-shift reconstruction work [24] extend adaptation approaches. Your “large population pretraining plus a few personal demonstrations” idea fits this active research area; the broad idea alone is not a new technical contribution.

### Sequence interfaces and foundation models

The generic neuromotor interface [25] demonstrates task-specific interaction at population scale. emg2qwerty [26] supplies a typing benchmark used by scaling/distillation work [27]. TinyMyo [28] and EMG-FM-Bench [29] explore broader pretraining. Foundation-model claims must still be checked on your sensor and causal, low-latency task.

These are selected citation paths, not a claim that every arrow is a direct reference in the preceding paper. Generic transformer, CLIP, masked-autoencoder, and hand-model references in the seed bibliographies explain implementation ancestry; they are not additional evidence that a wearable EMG product works.

## 5. Evidence by research direction

### A. Practical gesture decoding

| Work | Contribution and evidence | Implication for your band |
|---|---|---|
| Hudgins et al., 1993 [3] | Foundational multifunction myoelectric classification using signal patterns. | Implement time-domain features before choosing a large model. |
| Englehart & Hudgins, 2003 [4] | A real-time control scheme, rather than only isolated offline classification. | Decisions depend on windowing and controller behaviour as well as the classifier. |
| Atzori et al., 2016 [8] | CNN-based movement classification on established EMG datasets. | A CNN is a baseline, not by itself a differentiator. |
| Geng et al., 2016 [9] | Treats dense electrode arrays as spatial EMG images. | The spatial information in a high-density grid is absent from three sparse channels. Temporal voting must be audited for delay. |
| Du et al., 2017 [10] | Inter-session adaptation using deep domain adaptation. | Re-wearing is a recognised distribution-shift problem; same-session accuracy is insufficient. |
| Côté-Allard et al., 2019 [11] | Multi-user transfer learning for EMG gesture recognition. | Pretrain then personalize is established; measure the gain against training only on your own data. |
| EMGBench, 2024 [12] | A multi-dataset framework for cross-user generalization and adaptation. | Reuse its evaluation philosophy. A strong result on one curated dataset is not a general claim. |
| LibEMG, 2023– [30] | Open-source infrastructure for features, offline evaluation, and real-time EMG systems. | A useful implementation starting point; still verify compatibility with your BLE stream. |

**Recommended starting model:** per-channel RMS/MAV and waveform length, then optionally thresholded zero crossings and slope-sign changes; regularised LDA and an SVM comparison. Thresholds and scaling must be fitted on training data. Add a small causal 1D CNN or TCN only after establishing the baseline. This is a proposed experiment, not a promised outcome.

### B. Enrollment, re-wearing, and abstention

| Work | What it actually tests | Limitation to preserve |
|---|---|---|
| FS-HGR, 2021 [18] | Few-shot gesture recognition. | A “shot” should mean an independent demonstration, not another overlapping window from the same contraction. |
| Tam et al., 2024 [19] | Deep metric/meta learning, confidence, and rejection using high-density EMG. | Its 64-channel representation is not directly transferable to a three-channel band. |
| Scheme et al., 2013 [20] | Confidence-based rejection for improved myoelectric control. | Rejection trades coverage against mistakes; it does not guarantee no false commands. |
| Botros et al., 2025 [21] | Five gestures plus rest across 33 people, zero-/few-shot wrist and forearm decoding. Reported one-shot results improve substantially over zero-shot. | New-user adaptation is not the same as arbitrary new-class enrollment. Wrist outperformed forearm in that particular setup; placement superiority is not universal. |
| CPEP, 2025 [22] | Aligns EMG and pose embeddings; evaluates four in-distribution and four unseen gesture groups. | Zero-shot retrieval uses labelled pose references, including a test-pose retrieval bank in the reported formulation. It is not recognition of an arbitrary gesture from its name alone. |
| EMG-Adapt, 2026 [23] | Prototype-based and meta-learning adaptation across datasets and domains. | Inspect how each dataset defines a session; “cross-session” does not always mean removing and replacing the device on a later day. |
| MAST, 2024 preprint [24] | Masked reconstruction and transformer modelling for electrode shift in HD-EMG. | Dense-grid masking/shift augmentation does not faithfully simulate arbitrary movement of three sparse electrode pairs. |

CPEP deserves particular care: its two-second input windows and pose-reference retrieval protocol differ from a sub-200 ms consumer command interface. Table 2 reports unseen zero-shot macro accuracy around 48.1%, versus 75.7% in-distribution. Those are useful research results, not a solved arbitrary-gesture product. Its improvement percentages should not be mistaken for absolute accuracy or percentage-point gains. [22]

### C. Continuous pose, sequence decoding, and physical control

| Work | Approach | Interpretation |
|---|---|---|
| NeuroPose, 2021 [13] | Deep temporal pose reconstruction using a Myo wearable. The abstract reports 6.24° median joint-angle error and 101 ms smartphone processing. | A small-cohort result with a different sensor, metric, and evaluation. Processing time is not automatically end-to-end response time. |
| Sensing the Full Dynamics, 2022/2024 [14] | Dense neural-interface measurements and deep hand-kinematics modelling. | The sensing density is far beyond your band; it supports feasibility of the research direction, not a sparse-channel guarantee. |
| VQ-MyoPose, 2025 workshop [15] | Learns a discrete pose codebook, then predicts pose tokens from EMG. | Interesting structured output prior; its reported subset protocol must be matched before comparing with the full emg2pose benchmark. Pose-token learning is not raw-EMG clustering. |
| emg2tendon, 2025 [16] | Maps EMG to simulated musculoskeletal tendon-control representations using retargeting. | These are derived simulation targets, not direct measurements of human muscle forces. |
| DexEMG, 2026 preprint [17] | EMG-driven dexterous teleoperation and hand-pose estimation. | Relevant future direction; not proof of better robot-policy learning without a controlled physical-task evaluation. |
| Generic neuromotor interface, 2025 [25] | Population-scale wrist decoding for navigation, discrete gestures, and handwriting. Uses 16 bipolar channels; reports median handwriting around 20.9 words/minute. | Handwriting benefits from temporal/task structure. It does not establish full unrestricted pose reconstruction or arbitrary user gestures. |
| emg2qwerty, 2024 [26] | 108 users, 346.4 hours, bilateral wrist recordings paired with touch-typing targets. | Physical keyboard typing is not air typing. Useful for sequence modelling and representation experiments. |
| Scaling/distilling transformers, 2025 [27] | Scales and compresses typing decoders on emg2qwerty. | Compression results remain task- and hardware-dependent; benchmark the actual target MCU. |
| TinyMyo, 2025/2026 [28] | Masked pretraining across EMG data, with embedded inference measurements. | The final journal abstract differs from the preprint. Its 89 ms smaller-model compute result uses a one-second input on GAP9; it is not 89 ms end-to-end latency on ESP32. |
| EMG-FM-Bench, October 2026 preprint [29] | Benchmarks foundation-model transfer across multiple EMG datasets. | Abstract-screened here. Do not infer universal superiority or quote uninspected per-task results. |

For a three-to-four-second movement sequence, classification and pose estimation remain different options. A temporal recognizer can identify the sequence without estimating every joint. Start there if the product needs a command, and use explicit start/end or an activation gesture initially. Always document the delay introduced by waiting for the sequence to finish.

### D. How many channels? What does hardware research establish?

| Work | Evidence | Decision for you |
|---|---|---|
| Pelaez Murciego et al., 2022 [31] | Starts with dense acquisition, selects channels, and evaluates online control while wrist position changes. Eight individually selected channels outperform a generic circular arrangement in the reported online comparison. | Placement and posture robustness matter. This does not prove three channels suffice for your desired vocabulary. |
| Li et al., 2026 [32] | Evaluates 63 subsets from six upper-limb sites under a constrained rehabilitation-gesture protocol; reports strong three-/four-channel results. | Encouraging evidence for limited tasks, not all-finger tracking. Sensor quality, selected muscle sites, and task definition differ from yours. |
| Zhong et al., 2026 preprint [33] | Studies thumb decoding with different forearm electrode configurations. | Coverage, spacing, and montage matter as much as channel count. Do not generalize a thumb-specific placement result to all hand gestures. |
| Moin et al., 2021 [34] | Wearable sensing with adaptive embedded machine learning. | On-device adaptation has precedent; the contribution must be reliability, cost, power, or usability demonstrated on your hardware. |
| Soft wrist-worn system, 2023 [35] | Portable multichannel sEMG acquisition hardware. | Useful analogue/mechanical design reference. Acquisition quality is separate from recognition quality. |
| DLR thesis, 2025 [36] | Compares customizable electrode topologies and validates acquisition against reference measurements. | Helpful for later contact and common-mode-noise studies; not a ready-made gesture decoder. Only the thesis record/abstract was inspected here. |
| Mantilla et al., 2026 preprint [37] | Validation methodology for an EMG/movement wristband prototype. | Borrow the separation of acquisition, communication, and mechanical validation. Correlation against a reference is not a complete quality metric or certification. |

**There is no literature-supported universal ideal number of electrode pairs.** Three may support useful commands. More channels may improve separability and placement tolerance, but more contacts also complicate fit, electronics, bandwidth, and power. Decide from error analysis, not competitors’ photographs.

Do not buy additional hardware solely to increase a number. Upgrade when repeated measurements show a relevant gesture pair remains inseparable despite stable contact, appropriate placement, and sensible models—or when pose estimation becomes the explicit target. A higher-channel device with comparable acquisition can support a channel-ablation experiment. Wearing the same three-channel band in several positions sequentially does not create simultaneously measured multichannel data.

## 6. Datasets and code: what to use first

Public availability is not a commercial-use licence. Check data, model weights, code, and participant-use restrictions separately; a paper's open-access licence does not grant rights to its dataset. “Open source” applies to code more precisely than to human recordings.

| Resource | Suitable experiment | Important mismatch/access note |
|---|---|---|
| Your UDL recordings | Signal QA, causal control, re-wearing, enrollment, accidental commands | Most relevant deployment distribution; no established accuracy yet. Preserve raw decoded values and acquisition metadata. |
| NinaPro DB2 [6] | Conventional multiclass classification; channel ablation | 12 Delsys channels at 2 kHz, 40 intact participants; different sensors and placement. |
| NinaPro DB5 [7] | Wearable low-rate gesture baselines | Two Myo arrays on one forearm, 16 total channels at 200 Hz, 10 participants. This is not bilateral 16-channel wrist recording. |
| NinaPro DB6 [7] | Repeated-day/session evaluation | Ten participants, twice daily over five days. Stored arrays include unused channels; follow documentation. |
| CapgMyo / instantaneous-image work [9] | Spatial decoding and domain adaptation | High-density grids cannot be treated as equivalent to sparse contacts. |
| putEMG [38] | Gesture classification with another acquisition setup | Useful external validation; inspect session and task definitions before harmonizing labels. |
| emg2pose [1] | Temporal regression; pose supervision; channel ablations | Start with one session. Matching channel count after subsampling does not match physical sensor geometry. |
| EgoEMG [2] | EMG/vision/IMU ablations and multimodal representations | Pin paper/repository versions; confirm access and download terms. |
| emg2qwerty [26] | CTC-style sequence decoding; pretraining studies | Typing vocabulary, bilateral acquisition, and task context differ from commands. |
| MyoKi [39] | Activities of daily living and multimodal kinematics | Valuable for studying ordinary movement; inspect actual sensor layout and label coverage before importing. |
| LibEMG [30] | Feature extraction and evaluation tooling | Software framework, not a substitute for a properly split dataset. |
| Encord EMG-1 [40] | Industry direction for naturalistic EMG/vision data | Treat access, release version, licence, and claims separately. An announcement or signup page is not a locally inspected dataset. |

**Order for you:** your own data → a small NinaPro subset for pipeline sanity → DB6-style session evaluation → emg2pose for research on pose/representations. Do not spend the first week downloading every corpus.

## 7. How to read impressive results critically

1. **Split before windowing.** Keep complete repetitions, sessions, and users together. Neighbouring overlapping windows in train and test share signal content. A random window split can yield a flattering estimate of the wrong task.
2. **Name the generalization setting.** Same user/same session; same user/new session; new day with re-wearing; new user; new class; and new device are separate tests.
3. **Audit all learned preprocessing.** Normalisation, channel selection, thresholds, feature selection, and hyperparameters must not use held-out labels or future deployment data.
4. **Separate held contractions from transitions.** A method trained on clean plateaus can fail while entering or leaving a gesture. Include ordinary non-command movements, not just rest.
5. **Read the confusion matrix and per-user distribution.** Mean accuracy can hide a person who cannot use the system, a consistently missed gesture, or excessive rest predictions.
6. **Check causal availability.** Bidirectional networks, centred filters, whole-trial voting, and future signal windows can use information unavailable at the decision time. They may be valid offline models but need a streaming redesign.
7. **Keep latency components separate.** Evidence window, packet batching, BLE delivery, inference, smoothing, and host action all contribute. Fast inference alone is not fast interaction.
8. **Compare like metrics.** Mean versus median angle error, wrist-relative versus absolute fingertip coordinates, and initial-pose-assisted tracking versus regression are materially different.
9. **Treat interpolated labels correctly.** Upsampling motion capture does not create additional independent ground-truth measurements.
10. **Ablate motion sensors.** Compare EMG alone, accelerometer alone, and fusion. Otherwise a demo may be mostly an arm-motion detector.
11. **Look for closed-loop evidence.** Correct isolated windows do not establish useful command timing, comfortable control, or low accidental activation.
12. **Track licences and versions.** Paper, repository, checkpoint, and dataset can have different counts, metrics, and permissions.

These are review criteria, not blanket accusations that the papers above are invalid. A protocol can be appropriate for its research question yet unsuitable as evidence for your product claim.

## 8. What you should implement with the three-channel band

The following is an original experimental recommendation based on the gaps above, not a reproduction of one paper.

### Stage 1 — Establish trustworthy acquisition

- Record three channels separately, sample counters, host receive timestamps, firmware version, claimed sample rate, accelerometer values, and placement photographs/notes.
- Confirm the actual stream rate and packet-loss behaviour over several minutes. Host arrival intervals are not precise ADC sample timestamps.
- Save unmodified decoded samples. Derive filtered copies separately, with filter coefficients, sampling rate, units, and causal/noncausal mode recorded.
- Investigate baseline drift, mains contamination, clipping, contact changes, contraction repeatability, and movement artefacts. Do not infer an input microvolt scale from nominal ADC resolution and approximate gain alone.
- Your reported browser filter/sample-rate mismatch must be resolved before treating exported plots or processed CSVs as a research reference. Firmware settings and actual stream measurements should determine preprocessing.

### Stage 2 — A deliberately small recognizer

Choose three visibly distinct comfortable actions plus rest. Screen candidate gestures empirically; do not select them solely because their hand shapes differ. Collect approximately 20 independent repetitions per command over three sessions, including removal/re-wearing. Vary arm posture and effort in short blocks with breaks. Preserve transitions.

Compare:

| Experiment | Controlled comparison | Primary result |
|---|---|---|
| Baseline features | LDA versus SVM, same training trials | Macro F1 and confusion matrix |
| Evidence duration | 100, 150, 200 ms causal windows; fixed 20 ms update | Recognition/latency trade-off |
| Small temporal model | Causal CNN or TCN versus feature model | Gain on held-out sessions, not training accuracy |
| Sensor ablation | EMG versus accelerometer versus fusion | Whether muscle information adds value |
| Postprocessing | Confidence rejection, persistence, cooldown | False commands versus missed commands and delay |

A window length is not an exact mandatory delay after every onset: a continuously running model already has history. Nevertheless, the time required for enough discriminating post-onset evidence must be measured, especially for similar gestures.

### Stage 3 — Re-wearing and low-shot enrollment

Freeze a test session before tuning. Evaluate the next session with no adaptation, then with 1, 3, 5, and 10 independent demonstrations per gesture. Report the actual calibration time and include rest/non-command examples. Compare simple normalisation/prototype updates against full fine-tuning.

For **new-class enrollment**, hold a gesture out of initial model training, add it later, and test whether it confuses existing gestures. This is a separate experiment from giving a new user examples of existing classes. Expand from your own recordings to a small consenting pilot, then a larger cohort only when the collection protocol is stable. Five pilot users help find failures; they do not establish population-scale reliability.

### Stage 4 — Continuous negative testing

Record typing, walking, reaching, scratching, carrying objects, resting, and ordinary phone use. Measure false commands per hour, missed intended commands, and activation-to-action latency. Log both accepted and rejected events. An explicit activation state is a useful first interface choice.

Zero false commands in a short demo is weak evidence. Under a simple stationary Poisson model, zero events in T hours gives an approximate one-sided 95% upper rate of 3/T per hour: zero in 30 minutes is compatible with roughly six/hour; zero in ten hours with roughly 0.3/hour. Real use may violate that model, so also report users, activities, and exposure rather than only the bound.

### Stage 5 — Temporal gestures, then restricted pose

First classify two or three deliberate movement sequences, with clear onset/end definitions and causal processing. Do not jump directly to every finger joint. If pose remains important, collect synchronized video as imperfect training/reference labels and begin with a restricted continuous target, such as one finger's flexion or hand opening. Camera occlusion and synchronization uncertainty must be measured. Compare against a constant/previous-pose baseline, not only another neural network.

More channels become a decision after these experiments, not a prerequisite for starting them.

## 9. Research and product claims you can reasonably aim to make

| Claim | Evidence required |
|---|---|
| “Three-channel personalised control prototype” | Live commands, held-out repetitions, documented acquisition and host inference |
| “Fast enrollment” | Independent demonstrations, unseen-user/new-class protocols separated, measured enrollment time |
| “Works after re-wearing” | Multiple removals, days, placements, users, and specified adaptation budget |
| “Low false activations” | Hours of representative non-command activity and per-user event rates |
| “Works on-device” | Actual firmware, memory use, compute timing, energy and thermal measurements |
| “Tracks fingers continuously” | Synchronized reference kinematics, held-out sessions/users, error by joint and movement, causal latency |

The defensible early story is a reliable interaction system with a carefully measured deployment dataset. It is not “we are the first to combine ML and EMG,” “every hand gesture is distinguishable,” or “we match Meta with fewer electrodes.” A clear failure analysis and improvement over a strong baseline would be more credible than a large unqualified accuracy number.

## 10. Prioritised reading

1. **LibEMG and NinaPro documentation [5–7,30]:** implement a reproducible offline/live baseline.
2. **EMGBench [12] and the channel-selection paper [31]:** design splits and online evaluation before collecting too much data.
3. **emg2pose [1]:** understand temporal regression, supervision quality, and generalization settings.
4. **Botros et al., FS-HGR, and confidence rejection [18,20,21]:** turn enrollment and false activations into explicit experiments.
5. **CPEP and EgoEMG [2,22]:** investigate richer supervision after your measurement pipeline works.
6. **Meta's interface and emg2qwerty [25,26]:** understand population scale and task-specific temporal priors.
7. **TinyMyo/EMG-FM-Bench [28,29]:** evaluate pretraining later against your small models, with licences and compute measured.

## 11. Bibliography and access audit

Titles are linked to primary publications, author repositories, or official dataset pages. “Et al.” is used to avoid presenting unverified complete author lists. Venue/year refers to the stated version; preprint dates can differ from final publication dates.

### Seeds and established decoding

[1] Salter et al. (2024). **emg2pose: A Large and Diverse Benchmark for Surface Electromyographic Hand Pose Estimation.** NeurIPS Datasets and Benchmarks. [Paper](https://arxiv.org/html/2412.02725v1) · [Code/data tools](https://github.com/facebookresearch/emg2pose). **F/P.**

[2] Xi et al. (2026). **EgoEMG.** [Paper, arXiv:2605.05712](https://arxiv.org/html/2605.05712v1) · [Official repository](https://github.com/zhenqis123/EgoEMG). **F/P; preprint.** Dataset release counts differ from the paper. Repository code availability does not establish a commercial data licence; inspect separate release terms.

[3] Hudgins, Parker & Scott (1993). **A new strategy for multifunction myoelectric control.** IEEE Transactions on Biomedical Engineering. [PubMed](https://pubmed.ncbi.nlm.nih.gov/8468080/) · [DOI:10.1109/10.204774](https://doi.org/10.1109/10.204774). **A.**

[4] Englehart & Hudgins (2003). **A robust, real-time control scheme for multifunction myoelectric control.** IEEE Transactions on Biomedical Engineering. [PubMed](https://pubmed.ncbi.nlm.nih.gov/12848352/) · [DOI:10.1109/TBME.2003.813539](https://doi.org/10.1109/TBME.2003.813539). **A.**

[5] Atzori et al. (2014). **Electromyography data for non-invasive naturally-controlled robotic hand prostheses.** Scientific Data. [Paper](https://www.nature.com/articles/sdata201453) · [DOI:10.1038/sdata.2014.53](https://doi.org/10.1038/sdata.2014.53). **A/P.** Database releases have different sensor setups and preprocessing.

[6] NinaPro. **Database 2: acquisition and data documentation.** [Official DB2 page](https://ninapro.hevs.ch/instructions/DB2.html). **P.**

[7] NinaPro. **Database 5 and Database 6: acquisition and data documentation.** [DB5](https://ninapro.hevs.ch/instructions/DB5.html) · [DB6](https://ninapro.hevs.ch/instructions/DB6.html). **P.** Consult the cited original dataset papers and download conditions on each page.

[8] Atzori, Cognolato & Müller (2016). **Deep Learning with Convolutional Neural Networks Applied to Electromyography Data: A Resource for the Classification of Movements for Prosthetic Hands.** Frontiers in Neurorobotics. [Paper](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2016.00009/full). **A.**

[9] Geng et al. (2016). **Gesture recognition by instantaneous surface EMG images.** Scientific Reports. [Paper](https://www.nature.com/articles/srep36571). **A.**

[10] Du et al. (2017). **Surface EMG-Based Inter-Session Gesture Recognition Enhanced by Deep Domain Adaptation.** Sensors 17(3),458. [Paper](https://www.mdpi.com/1424-8220/17/3/458). **A.** Related author work: [Semi-Supervised Learning for Surface EMG-based Gesture Recognition, IJCAI 2017](https://www.ijcai.org/proceedings/2017/225).

[11] Côté-Allard et al. (2019). **Deep Learning for Electromyographic Hand Gesture Signal Classification Using Transfer Learning.** IEEE Transactions on Neural Systems and Rehabilitation Engineering. [PubMed](https://pubmed.ncbi.nlm.nih.gov/30714928/) · [Author manuscript](https://graal.ift.ulaval.ca/ulysse/articles/transfer_learning_journal_paper.pdf). **A.**

[12] Yang et al. (2024). **EMGBench.** NeurIPS Datasets and Benchmarks. [Official project](https://emgbench.github.io/) · [Paper, arXiv:2410.23625](https://arxiv.org/abs/2410.23625). **A/P.**

### Pose and physical control

[13] Liu, Zhang & Gowda (2021). **NeuroPose: 3D Hand Pose Tracking Using EMG Wearables.** The Web Conference. [Author PDF](https://www.cse.psu.edu/~mkg31/papers/Neuropose_www.pdf) · [DOI:10.1145/3442381.3449890](https://doi.org/10.1145/3442381.3449890). **A, author manuscript sections consulted.**

[14] Sîmpetru et al. **Sensing the Full Dynamics of the Human Hand with a Neural Interface and Deep Learning.** [Official project and paper link](https://raulsimpetru.github.io/Sensing-the-Full-Dynamics-of-the-Human-Hand-with-a-Neural-Interface-and-Deep-Learning/) · [2022 preprint DOI](https://doi.org/10.1101/2022.07.29.502064). **P; discussed through emg2pose.** Original preprint access was blocked in this review; no complete full-text appraisal is claimed.

[15] Lovecchio et al. (2025). **VQ-MyoPose.** HANDS workshop, ICCV. [Author/workshop PDF](https://hands-workshop.org/files/2025/VQ-MyoPose.pdf). **A, opening paper sections inspected.**

[16] Verma et al. (2025). **emg2tendon.** Robotics: Science and Systems. [Proceedings](https://roboticsproceedings.org/rss21/p135.html) · [Official project](https://emg2tendon.github.io/). **A/P.**

[17] Zhao et al. (2026). **DexEMG.** [arXiv:2603.05861](https://arxiv.org/abs/2603.05861). **A; preprint.**

### Few-shot, adaptation, and rejection

[18] Rahimian et al. (2021). **FS-HGR: Few-shot Learning for Hand Gesture Recognition via Electromyography.** IEEE Transactions on Neural Systems and Rehabilitation Engineering. [Preprint](https://arxiv.org/abs/2011.06104) · [Code](https://github.com/EllaRahimian/FS-HGR) · [DOI:10.1109/TNSRE.2021.3077413](https://doi.org/10.1109/TNSRE.2021.3077413). **A/P.**

[19] Tam et al. (2024). **Towards Robust and Interpretable EMG-based Hand Gesture Recognition using Deep Metric Meta Learning.** [arXiv:2404.15360](https://arxiv.org/html/2404.15360v1). **F; preprint version inspected.**

[20] Scheme et al. (2013). **Confidence-based rejection for improved pattern recognition myoelectric control.** IEEE Transactions on Biomedical Engineering. [PubMed](https://pubmed.ncbi.nlm.nih.gov/23322756/). **A.**

[21] Botros et al. (2025). **From zero- to few-shot: deep temporal learning of wrist EMG enables scalable cross-user gesture recognition.** Journal of Neural Engineering. [PubMed](https://pubmed.ncbi.nlm.nih.gov/40967242/) · [DOI:10.1088/1741-2552/ae08eb](https://doi.org/10.1088/1741-2552/ae08eb). **A.**

[22] Cui et al. (2025). **CPEP: Contrastive Pose-EMG Pre-training Enhances Gesture Generalization on EMG Signals.** Foundation Models for the Brain and Body workshop. [arXiv:2509.04699v3](https://arxiv.org/html/2509.04699v3). **F.** Check the pose-retrieval bank and two-second evaluation windows before adopting its zero-shot claims.

[23] Lee et al. (2026). **EMG-Adapt.** Scientific Reports. [Full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC13086972/) · [DOI:10.1038/s41598-026-40352-6](https://doi.org/10.1038/s41598-026-40352-6). **F.**

[24] Laamerad et al. (2024). **Masked Autoencoder with Swin Transformer Network for Mitigating Electrode Shift in HD-EMG-based Gesture Recognition.** [arXiv:2410.17261](https://arxiv.org/abs/2410.17261). **A; preprint.**

### Population interfaces and pretraining

[25] Meta/CTRL-labs researchers (2025). **A generic non-invasive neuromotor interface for human-computer interaction.** Nature. [Paper](https://www.nature.com/articles/s41586-025-09255-w) · [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12443603/) · [Data/code repository](https://github.com/facebookresearch/generic-neuromotor-interface-data). **A/P, indexed methods/results consulted; full Nature/PMC access restricted during this review.**

[26] Sivakumar et al. (2024). **emg2qwerty: A Large Dataset with Baselines for Touch Typing using Surface Electromyography.** NeurIPS Datasets and Benchmarks. [arXiv:2410.20081](https://arxiv.org/abs/2410.20081) · [Official repository](https://github.com/facebookresearch/emg2qwerty). **A/P.** Some later references use a 2025 version date.

[27] Mehlman et al. (2025). **Scaling and Distilling Transformer Models for sEMG.** Transactions on Machine Learning Research. [arXiv:2507.22094](https://arxiv.org/abs/2507.22094) · [OpenReview PDF](https://openreview.net/pdf?id=hFPWThwUiZ). **A.**

[28] Fasulo et al. (2025 preprint; 2026 journal). **TinyMyo.** [Preprint](https://arxiv.org/html/2512.15729v1) · [Journal record](https://pubmed.ncbi.nlm.nih.gov/42785278/) · [DOI:10.1109/TBME.2026.3736157](https://doi.org/10.1109/TBME.2026.3736157) · [BioFoundation code](https://github.com/pulp-bio/BioFoundation). **F for preprint sections; A for final journal record.** Do not combine old and new result numbers.

[29] Wu et al. (2026). **EMG-FM-Bench.** [arXiv:2610.06450](https://arxiv.org/abs/2610.06450). **A; recent preprint, abstract screened through search indexing; full-text retrieval unsuccessful.**

### Tools, channels, and hardware

[30] LibEMG (2023–). **An open-source library for electromyography research.** [Documentation](https://libemg.github.io/libemg/) · [Repository](https://github.com/LibEMG/libemg) · [DOI:10.1109/ACCESS.2023.3304544](https://doi.org/10.1109/ACCESS.2023.3304544). **P.**

[31] Pelaez Murciego et al. (2022). **Reducing the number of EMG electrodes during online hand gesture classification with changing wrist positions.** Journal of NeuroEngineering and Rehabilitation. [Full text](https://link.springer.com/article/10.1186/s12984-022-01056-w). **F.**

[32] Li et al. (2026). **How few channels suffice? A hierarchical validation framework for minimal-channel wearable sEMG in upper-limb rehabilitation gesture recognition.** Journal of NeuroEngineering and Rehabilitation. [Full text](https://link.springer.com/article/10.1186/s12984-026-02039-x). **F, substantive sections inspected.**

[33] Zhong, Ma & Nazarpour (2026). **Electrode-configuration study for thumb-movement classification.** [arXiv:2604.04623v1](https://arxiv.org/html/2604.04623v1). **F, substantive sections inspected; preprint.** Descriptive citation label used here rather than an unverified exact title.

[34] Moin et al. (2021). **A wearable biosensing system with in-sensor adaptive machine learning for hand gesture recognition.** Nature Electronics. [Paper](https://www.nature.com/articles/s41928-020-00510-8). **A.** Published online in 2020; journal volume dated 2021.

[35] (2023). **A Wireless, High-Quality, Soft and Portable Wrist-Worn System for sEMG Signal Detection.** Micromachines 14(5),1085. [Paper](https://www.mdpi.com/2072-666X/14/5/1085) · [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10221799/). **A.** This is the 2023 hardware paper previously shared in the project.

[36] Große-Westermann (2025). **Design and Validation of a Customizable Electromyographic Sensor Module for the DLR Research Wristband.** Master's thesis. [DLR record and PDF](https://elib.dlr.de/217281/). **A only; 89-page thesis not read in full.**

[37] Mantilla et al. (2026). **Design of a validation methodology for a prototype wristband for capturing muscle signals and upper limb movement.** [arXiv:2605.11148](https://arxiv.org/abs/2605.11148). **A; preprint.**

[38] Kaczmarek et al. (2019). **putEMG—A Surface Electromyography Hand Gesture Recognition Dataset.** [Full-text source](https://pmc.ncbi.nlm.nih.gov/articles/PMC6720505/). **A.**

[39] Andreas et al. (2025). **MyoKi dataset.** Scientific Data. [Paper](https://www.nature.com/articles/s41597-025-05852-6) · [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12394619/) · [DLR](https://elib.dlr.de/221692/). **A/P.** Uses EMG and kinematic/multimodal measurements during activities of daily living; inspect release documentation for exact subsets.

[40] Encord Data Labs. **EMG-1 technical announcement.** [Official page](https://encord.com/data-labs/introducing-emg-1/). **Industry context, not peer-reviewed performance evidence.** No raw data or licence was inspected locally for this review.

### Additional screened reading

[41] Hu et al. (2018). **A novel attention-based hybrid CNN-RNN architecture for sEMG-based gesture recognition.** [Full-text source](https://pmc.ncbi.nlm.nih.gov/articles/PMC6207326/). **A.** Temporal/attention architecture precedent.

[42] Raghu et al. (2024/2025). **VICReg-based work on continuous/dynamic EMG recognition.** [arXiv:2409.11632](https://arxiv.org/abs/2409.11632) · [DOI:10.1016/j.compbiomed.2024.109479](https://doi.org/10.1016/j.compbiomed.2024.109479). **A.** Useful reminder that gains on continuous transitions and on held/ramp contractions can differ.

[43] Côté-Allard et al. (2019). **A Low-Cost, Wireless, 3-D-Printed Custom Armband for sEMG Hand Gesture Recognition.** Sensors 19(12),2811. [Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6631507/). **A.** Wearable acquisition precedent; not a performance target for your hardware.

### Excluded and unresolved evidence

- **arXiv:2605.03462**, currently titled *From Muscle Bursts to Motor Intent: Self-Supervised Token Modeling for Heterogeneous EMG*, is [withdrawn](https://arxiv.org/abs/2605.03462). The current notice cites unresolved reliability/reproducibility and dataset-release/validation issues. Older search indexing may show a different AEMG title. Its performance claims are excluded from this review's recommendations.
- Some seed-paper bibliography entries have author/metadata inconsistencies. Original publication records were used where available rather than copying author lists blindly. This is a bibliographic caution, not a judgement about experimental validity.
- The DLR thesis, some paywalled original articles, and the newest benchmark were not fully appraised. These are explicitly marked above rather than treated as fully verified results.
- Company demonstrations from Axonnic, Hemlock, Preload, and Mudra are not used here as controlled scientific comparisons. Their current proprietary channel layouts, training data, or decoder architecture cannot be inferred reliably from videos.

## 12. Next experiment to log

**LIT-01 — Literature review completed; acquisition-to-control baseline planned.**

Inputs: this review, three-channel UDL forearm band, existing raw-recorder work. Next action: acquire a small labelled three-gesture/rest dataset across separate wearings, run a feature/LDA baseline with trial/session splits, then test accelerometer-only and EMG-only variants. Preserve the dataset, configuration, split manifest, confusion matrix, false-command log, and measured latency. No recognition improvement is claimed until those results exist.
