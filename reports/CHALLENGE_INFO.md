# Challenge facts collected from public sources (2026-10-15)

Sources:
* challenge site: https://sites.google.com/view/abc2027-uwbfalltrack-challenge (The Challenge, Registration, Dataset and Tutorial, FAQ, Organisers)
* ABC 2027 main site: https://autocare.ai/abc2027
* package README and starter notebook
* organisers' answers to our Q1/Q2

## Important dates (ABC 2027 site, "ABC 2027 Challenges"; all deadlines 23:59 AoE)

| item | date |
|---|---|
| Tutorial / dataset open | Oct 13, 2026 |
| **Test data sent to participants** | **Oct 20, 2026** |
| **Challenge registration closes** | **Oct 27, 2026** |
| **Submission of results** | **Nov 3, 2026** |
| Submission of papers | Nov 10, 2026 |
| Reviews sent to participants | Nov 17, 2026 |
| Camera-ready papers | Dec 8, 2026 |
| Conference (Sydney, hybrid) | Jan 19–22, 2027 |

Registration: Google form linked on the challenge site (title = team name; abstract may be blank; choose the track).

## Rules and requirements (Registration page and README)

* No limit on team size. Only supervisors may be in several teams. Students of the host/organising labs are not eligible.
* Private sharing of code or data outside the team is prohibited.
* Using the source LIARA recordings (the original corpora or any copy) is prohibited for training and inference.
  Top-ranked teams must provide code and models for verification.
* Q2 (organisers): no matching, comparing or aligning content across test streams, within a stream, or with train/val
  segments to detect repeated portions. Each test stream is processed independently; falls must be detected from the signal.
* Q1 (organisers): the test streams use the same generator and settings as train/val; only the seed and the participants differ.
* The package's own starter notebook builds an extra stream from `segments/` with the official generator, so using the
  shipped segments that way is demonstrated by the organisers. Our final models are trained only on the published train
  and val streams; regenerated streams are used only for validation (SR-CV).

## Metric (README)

* Ranking = mean event F1 at tIoU 0.1 / 0.3 / 0.5 (`--ignore-flagged`).
* Tie-breaks: lower median |start error| at θ0.3, then fewer false alarms/hour at θ0.3.
* `t_alarm` is optional and only used for the reported latency, which does not break ties.

## Paper

* English, ABC format, 8–12 pages including references; published in IJABC (J-Stage).
* Sections: abstract, methodology (dataset, preprocessing, model, evaluation metrics), results and discussion,
  conclusion, references.
* Must cite three main papers: the dataset paper, the LOPDO evaluation, and the challenge summary paper (provided by the hosts).
* At least one team member must present at the workshop. Registration fee: student 290 USD / regular 390 USD
  (IJABC/Challenge/Position).
* Prize: 100,000 JPY for the winning team; participation certificate for every team.

## Still not answered publicly (ask the organisers)

1. Within one stream, may frames identical to the immediately preceding frame of the same radar (50 Hz alignment) be
   dropped before computing features?
2. May the 0.2 s crossfade junctions detected from a stream's own signal be used as context for detection?
3. May the package's annotation code (`annotate.py`, `propose_fall`) be run at inference to refine start/end times?
4. How many result submissions are allowed, and which one counts?
