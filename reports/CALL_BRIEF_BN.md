# Sir er sathe call: A to Z (Banglish)

Status: 2026-10-17 (round 4 sesh). Shob number repo r report theke (`notion_export/UWB-FallTrack Challenge - A to Z.md`).

---

## 0. Prothom 30 second (eta diye shuru korben)

"Sir, amader system ta validation e **0.937** ar subject-disjoint cross-validation e **0.914** score korche. Official
baseline er score chilo **0.315**. Test data ashbe **Oct 20**, result jabe ekbar-i, **Nov 3** er moddhe. Code package ready,
test kora. Test e amra **0.90–0.91** expect korchi, karon CV te unseen subject diye test kora hoyeche, test o temon."

---

## 1. Problem statement

- Kaj: lomba, continuous radar stream e protita **fall** khuje ber kora, ar tar **start time ar end time** bola.
- Sensor: **3 ta UWB radar** (XeThru X4M200). Protita frame e radar prati 368 ta value, **50 Hz**.
- Output: protita fall er jonno CSV te ek row: `stream_id, t_start, t_end, score, t_alarm`.
- Rule:
  - shudhu challenge package er data use kora jabe; LIARA er original recording nishiddho;
  - protita test stream alada alada process korte hobe.
- Organiser: UQAC (Bouchard lab, LIARA) ar Kyutech (Sozo Inoue lab), ABC 2027 conference.

## 2. Score kivabe hoy, deadline

- Ranking score: **tIoU 0.1, 0.3, 0.5 e event F1 er gor (mean)**. Flagged (kharap quality) fall gula ignore hoy.
- Tie hole: (1) median start error kom kar, (2) ghontay false alarm kom kar.
- **Result ekbar-i submit kora jay.** Shob team ke code o dite hoy.
- Paper: 8–12 page, IJABC, **Nov 10**. Dataset, LOPDO ar challenge summary paper cite korte hobe. Prize 100,000 JPY.

| Ki | Kobe |
| --- | --- |
| Test data | Oct 20 |
| Registration sesh | Oct 27 |
| **Result (ekbar)** | **Nov 3** |
| Paper | Nov 10 |
| Conference (Sydney, hybrid) | Jan 19–22, 2027 |

## 3. Dataset

| | Train | Val |
| --- | --- | --- |
| Stream | 13 (6.18 ghonta) | 5 (2.05 ghonta) |
| Annotated fall | 359 | 119 |
| Score e count hoy (flagged na) | 186 | 67 |

Gurutto purno: stream gula **asol continuous recording na**. Fall clip ar daily-activity (ADL) clip alada recording
theke kete, shuffle kore, **0.2 s crossfade** diye jora lagano. Majhe majhe "other activity" ar khali room er recording
(77–120 s) bosano.

## 4. Data analysis: ki ki pailam (main part, ektu shomoy niye bolben)

1. **Frame format.** 368 value asole **184 ta complex range bin: I = 0–183, Q = 184–367**. Tin bhabe check korsi:
   range-time map 184 e bhenge jay; dui half er motion profile r ≈ 0.87–0.92 correlate kore; LIARA paper e 184 bin,
   5.14 cm. Official baseline egula ke 368 ta real number dhore.
2. **Fall er timing.** Junction (dui clip er jora) er median **3.0 s** pore fall shuru hoy, fall median **1.74 s** lomba.
   Tarpor manush 3–10 s shuye thake. Ghontay prai **58 ta fall**.
3. **Baseline er bhul kothay.** Prati jora te baseline er slow filter prai **55 gun** spike kore.
   **Baseline er 90% false alarm junction er 1 s er moddhe.** Junction nijei khub shohoje dhora jay (AUC 0.98–0.99).
4. **Asol fall er signature.** Frame-difference energy fall er bhitore peak kore (87–90% fall e, ADL er 500–1500 gun).
   Fall er **age vs pore static profile** er change diye fall ar ADL alada kora jay (AUC ~0.98–0.99).
   Protita mat position mainly ek radar dekhe (D1 → R29444, D2 → R30722, D3 → R29171).
5. **Aro kichu.**
   - Fall ar ADL alada session e record kora, tai 1 second background dekhei alada kora jay (AUC 0.94–0.98).
   - **23–27% frame ager frame er hubohu copy** (sample-and-hold), tai asol rate ~37 Hz.
   - **Label gula algorithm diye banano**: package er `annotate.propose_fall` abar chalale start/end 0.02/0.04 s er
     moddhe mile jay.
   - Dataset 3 (D3) er prai shob fall flagged, tai score mainly D1 ar D2 er upor.
6. **Fall er aage kon activity?** Kono specific na. Chi-square p = 0.49, karon generator clip shuffle kore.

Website/README er sathe data er je gula mile na: frame layout (complex), frame rate (repeat), stream structure (shuffled
clip), D3 (prai shob ignore), label (algorithm, motion peak na).

## 5. Amra kivabe kaj korsi (process)

1. Data analysis + ekta prototype diye pattern gula prove korsi.
2. Protita finding alada code diye abar check korsi; literature ar rule er sathe plan milaisi.
3. **Subject-disjoint CV (SR-CV)** banaisi: package er generator diye unseen subject er stream banaye 4 fold, 253 ta
   evaluable fall. Eta test er moto, tai eta-i main judge.
4. **Promotion rule** (kono change rakhbo tokhoni jokhon tin tai thik):
   - paired subject-cluster bootstrap e P(gain) ≥ 0.90;
   - 4 fold er moddhe kompokkhe 3 fold e improve;
   - val 0.02 er beshi kome na.
5. Prottek change kompokkhe **3 ta alada TCN seed set** e test korsi, karon ek seed e "win" pore ulte jay.

## 6. Final system (`final_norep`)

CPU te ek stream er jonno ek minute er kom lage, shudhu open-source library (scikit-learn, LightGBM, PyTorch).

1. **Feature** prati 0.2 s e, 112 channel: 64 ta physics/structure (motion, frame-difference energy, static change,
   junction timing, segment statistics, background fingerprint) + 48 ta range-band energy.
2. **Dui ta detector:** gradient boosting (HGB) ar dilated temporal CNN (TCN, 3 seed).
   Fall probability = 0.75 × TCN + 0.25 × HGB.
3. **Segment gate:** LightGBM protita junction-to-junction segment ke rate kore; score = p × (0.7 + 0.3 × P(fall segment)).
4. **Decoding:** threshold, smoothing, minimum duration/gap; shob shudhu training er out-of-fold prediction e tune.
5. **Boundary:** package er annotator ke detection er charpashe 6 ta crop e abar chalai; je crop gula mile (tIoU ≥ 0.3)
   tader median. Ete start error 0.22 s theke 0.04–0.06 s e neme ashe.
6. **Clean-up:** 4 s time NMS, tarpor official validator. Test data alada mone hole drift monitor warning dey.

## 7. Score kivabe barlo (step by step)

| Step | Val |
| --- | --- |
| Official baseline | 0.315 |
| Gradient boosting + physics feature | 0.790 |
| + segment structure (junction timing) | 0.876 |
| + background fingerprint | 0.884 |
| + annotator replay (boundary) | 0.928 |
| TCN (3 seed) + HGB blend | 0.930 / 0.934 |
| Blend + annotator replay | 0.965 (CV 0.886) |
| + replay gate 0.3, segment gate, 4 s NMS → `final` | 0.952 (CV 0.918) |
| Rule-safe kore repeat feature bad → **`final_norep`** | **0.937 (CV 0.914)** |

- Val ekai shob na: shesher step gula val e olpo komaise kintu CV (unseen subject) barache, ar amra CV diye decide kori.
- Segment gate + NMS ekai CV te +0.028 dise, false alarm 35.7 → 20.0.

## 8. Result

| Data | Ranking F1 | TP / FP / FN @0.3 | Precision | Recall |
| --- | --- | --- | --- | --- |
| Training (13 stream, out-of-fold) | 0.941 | 176 / 5 / 10 | 97.2% | 94.6% |
| **Validation (5 stream)** | **0.937** | 63 / 3 / 4 | 95.5% | 94.0% |
| **Subject-disjoint CV (test er moto)** | **0.914** | 239 / 23 / 14 | 91.2% | 94.5% |

- Training ar val prai shoman, tai **overfitting nai**.
- Median start error 0.04–0.05 s; ghontay false alarm 1–1.5.
- Accuracy bola hoy na, karon 97% er beshi shomoy e kono fall nai; tai event F1.

## 9. Ablation (kon part koto dorkar)

3 seed set, GPU te. Base 0.908.

| Bad dile | Score kome | Mane |
| --- | --- | --- |
| TCN | −0.060 | shobcheye dorkari |
| Annotator replay | −0.014 (start error 0.04 → 0.32 s) | dorkari |
| 4 s NMS | −0.013 (4/4 fold) | dorkari |
| HGB / segment gate | −0.006 kore | olpo help kore |
| Range band / fingerprint | noise er moddhe | neutral |

## 10. Je gula try korsi kintu rakhini (sir jiggesh korte paren)

- Complex I/Q phase feature: val e confirm hoyni.
- Ensemble voting (per-radar model, stacking, box fusion): simple average er theke bhalo na.
- Missed fall recover: +0.0003, kaj kore na. 14 ta miss er 8 ta te row ache kintu bhul jaygay.
- Learned boundary head (TriDet-style): annotator replay er theke kharap.
- Image verifier (open-source ResNet-18, range-time image): +0.013 kintu shudhu 1/4 fold e.
- Boundary blend: +0.006 kintu fresh seed e 2/4 fold, ar start error barache, tai rakhini.
- Range band bad: prothome +0.009, fresh seed e −0.011, mane noise chilo.
- **Cross-stream ADL duplicate mask: rule bhange, tai kokhono use kori nai.**

## 11. Round 4 (sesh)

- 9 TCN seed (3 er bodole): +0.006, P 0.87 → pass na.
- **Chute04 segment feature** (chair theke fall er jonno segment classifier e 6 ta feature beshi):
  - seed set 6–8 e +0.005, P 0.906, 3/4 fold → boundary te pass;
  - tai 6 ta seed set (3–8) eksathe check korsi: +0.0024, P 0.84, 2/4 fold → **pass na**.
- **Decision: kono change nai, `final_norep` ei submission.** Mane amra rule mene cholchi, luck e asha "win" nicchi na.
- Sir ke bolar moto line: "Notun idea gula amra fresh seed e abar test kori. Ekta idea 3 seed e pass korsilo, 6 seed e
  korenai, tai rakhini."

## 12. Rule compliance (sir eta niye proshno korte paren)

| Variant | Repeat feature | Annotator er aage repeated frame drop | CV | Val |
| --- | --- | --- | --- | --- |
| `final` | Ha | Ha | 0.918 | 0.952 |
| **`final_norep`** (submission) | Na | Ha (permission chaoa hoyeche) | 0.914 | 0.937 |
| `final_strict` (backup) | Na | Na | 0.896 | 0.937 |

**Q3 er por decision: `final_norep` submit korbo.**
- Dedup ta shudhu same radar er **thik ager frame** er sathe mile dekhe; 50 Hz sample-and-hold ta undo kore. Eta kono
  repeated activity khoje na, tai Q2 je jinish nishiddho korse (repeated portion detect kora) seta na.
- Labh: CV te +0.018, 4/4 fold e. Val e dedup thakuk ba na thakuk 0.937.
- Risk komano: code package er README te likha ase, paper e o likhbo, ar organiser er Q3 reply ta proof hishebe
  rakhbo. `final_strict` ready, ek command e switch kora jay.

Shob variant e: protita stream alada, kono cross-stream matching nai, shudhu package data, LIARA recording nai, shudhu
open-source library.

Organiser er answer:
- Q1: test same generator ar setting e, shudhu seed ar participant alada → tai CV ke test proxy dhorsi.
- Q2: kono repeated content matching chalbe na, protita stream alada → cross-stream mask off.
- **Q3:** ek stream er bhitore same radar er ager frame er hubohu copy drop kora jabe kina. Answer: "It's a competition,
  so you have to decide by yourself." → amra `final_norep` rakhsi (niche decision dekhun).
  Ha hole `final_norep`, na hole ba answer na ashle `final_strict`.

## 13. Release day plan

```bash
git pull
make predict-test VARIANT=final_norep TEST_DIR=/path/to/streams/test   # CPU
make package VARIANT=final_norep
```

- `predict-test` model er sha256 check kore, prediction kore, official validator ar drift monitor chalay.
- `package` code + model + run script + exact library version + README + hash diye zip banay.
- Test kora: notun folder e unzip kore chalale dry-run er CSV byte by byte same ashe.

## 14. Related work (paper er jonno)

- LIARA dataset er paper gula (Maitre et al. JBHI 2021; Imbeault-Nepton et al. GoodIT 2023; Abudalfa & Bouchard 2025):
  shob **age theke kata clip classify** kore, keu time e fall localise kore na.
- Kachakachi method: sensor e temporal action localisation (Bock et al. IMWUT 2024; XRF V2 IMWUT 2025; WiFiTAD AAAI 2025).
- Radar e fall **localisation** er tIoU-based event F1 result amra published pai nai. Amader 0.91–0.95 related work er
  shoman ba upore.

## 15. Sir jodi jiggesh koren (short answer)

- **"98–99% hobe na kno?"** Label algorithm diye banano, D3 prai shob ignore, ar tIoU 0.5 e boundary khub tight. Unseen
  subject e 0.91 realistic; related work e eta-i top level.
- **"Overfit kina?"** Na: training OOF 0.941, val 0.937, ar unseen subject CV 0.914.
- **"Test e koto ashbe?"** 0.90–0.91 expect korchi.
- **"Rule bhangar risk?"** Q2 answer onujayi cross-stream kichu nai. Q3 te organiser decision amader upor chhere dise;
  amra dedup rakhsi ar README/paper e khule bolsi. Backup
  `final_strict` ready.
- **"GPU lage?"** Na, submission CPU te, shipped model diye, sha256 check kore.
- **"Ekhon baki ki?"** Registration confirm (Oct 27 er aage), Oct 20 e test predict, Nov 3 er
  moddhe submit, Nov 10 er moddhe paper.

## 16. Sir er kache ja chaibo

1. Q3 decision (`final_norep`) e sir er OK.
2. Registration confirm kora (Oct 27 er aage).
3. Paper er author order ar ke workshop e present korbe.
