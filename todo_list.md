# Weekly to-do

One list per week and per person, following the timeline of the proposal.

- **Erulan** does all the downloads and all the runs on his desktop GPU, and nothing else.
- **Jue** writes the training code and the model part of the report.
- **Elise** writes the evaluation code, sends the applications, interprets the results and writes the slides and the report. She can run light tests herself (less than an hour on her own GPU).

So Jue and Elise write the scripts and test them on a few files, and Erulan runs them on everything and sends back the outputs.

## Week 1: 14 to 17 Nov. Prototype and interim presentation

Goal: YourMT3+ runs on our data, and we present the first scores on 17 Nov.

**Erulan (downloads and runs)**

- [ ] Download CSD, N20EMv2 and ISMIR2014
- [ ] Install YourMT3+ and download its checkpoints
- [ ] Run YourMT3+ on the CSD test recordings (English 41 to 50 and Korean) and send the MIDI files to Elise
- [ ] Note the latency and the GPU memory of that run
- [ ] If there is time: try to overfit 20 clips with Jue's instructions, to see if fine-tuning runs at all

**Jue (training code)**

- [X] Read `train.py` and the dataset config, and write down what a new dataset needs
- [X] Write the instructions for Erulan's overfit test
- [X] Write the list of training and test files for CSD (English 1 to 40, English 41 to 50, Korean)
- [ ] Slides: the model, H1 and H2, the timeline

**Elise (evaluation and presentation)**

- [ ] Send the applications for M4Singer, Opencpop and TONAS
- [ ] Write the scoring script with `mir_eval` (COn, COnP, COnPOff F1) and test it on a few files
- [ ] Write the script that counts the error types: wrong pitch, octave error, note cut in several, notes merged, onset more than 50 ms off
- [ ] Score Erulan's MIDI files and pick 3 typical failures as piano-roll pictures
- [ ] Slides: the problem with the demo, the data, the baseline scores and the failures

**Everyone**

- [ ] Agree on the final proposal
- [ ] Rehearse the presentation once, in our own words

## Week 2: 18 to 24 Nov. First fine-tune

Goal: one LoRA run with scores, and the go/no-go on 24 Nov.

**Erulan**

- [ ] Run the baseline on all 4 test sets, and again with a smaller YourMT3+ checkpoint
- [ ] Run the baseline in FP16 and with batch sizes 1, 8 and 32, and note the speed
- [ ] Run the first LoRA fine-tune (rank 16) on CSD only, then transcribe the test sets with it
- [ ] Send 10 GTSinger clips with their notes to Elise

**Jue**

- [ ] Write the code that puts CSD and N20EMv2 in the YourMT3+ training format
- [ ] Write the code that adds LoRA to the attention layers
- [ ] Write the download script, so Erulan's setup can be reproduced

**Elise**

- [ ] Write the label converters for N20EMv2 and ISMIR2014
- [ ] Split N20EMv2 by singer into training and test
- [ ] Score the runs and fill the first rows of the results matrix
- [ ] Listen to the 10 GTSinger clips and decide if we use it
- [ ] Repo structure and README skeleton

**Everyone**

- [ ] 24 Nov: go/no-go. If fine-tuning YourMT3+ does not work, we switch to the Whisper fallback

## Week 3: 25 Nov to 1 Dec. All fine-tuning variants

Goal: the H1 rows of the matrix. Quiz 3 is on 1 Dec, so this week is lighter.

**Erulan**

- [ ] Run LoRA rank 4 and rank 16 on the full training set
- [ ] Run one full fine-tune
- [ ] Note the training time of each run
- [ ] Transcribe the 4 test sets with each model and send the MIDI files

**Jue**

- [ ] Write the code that merges LoRA into the weights
- [ ] Write the code for INT8 quantization
- [ ] Count the trained parameters of each variant

**Elise**

- [ ] Score every new model on the 4 test sets
- [ ] Count the errors before and after fine-tuning
- [ ] Pick 2 examples that fine-tuning fixed and 1 that it did not

## Week 4: 2 to 8 Dec. Inference optimization and full matrix

Goal: every cell of the results matrix is filled.

**Erulan**

- [ ] Run the best model with LoRA merged, in FP16 and in INT8
- [ ] Measure latency, memory and throughput for every variant
- [ ] Re-run anything that looks wrong

**Elise**

- [ ] Fill and check the whole results matrix
- [ ] Make the plot of quality against latency, and the result tables
- [ ] Start writing motivation, data and evaluation

**Jue**

- [ ] Write the model, fine-tuning and inference optimization sections
- [ ] Draft the summary figure

**Everyone**

- [ ] Only if the matrix is full: pick one item from "Optional extras"

## Week 5: 9 to 15 Dec. Report and presentation

Goal: report uploaded and presentation ready. Check the exact deadline on ILIAS.

**Elise**

- [ ] Report: motivation, prior art, data, evaluation, conclusion

**Jue**

- [ ] Report: model, fine-tuning, inference optimization, summary figure
- [ ] Finish the README

**Erulan**

- [ ] Last re-runs if a number is missing
- [ ] Check that the repo runs from a clean install

**Everyone**

- [ ] Report is 7 to 8 pages in the ICML template, with the repo link
- [ ] 20-minute presentation, rehearsed once
- [ ] Interview preparation: everyone can explain the whole pipeline



---


## Feedback from teacher

**Title:** Improvement of vocal to MIDI transcription models

**Problem:** create an audio to MIDI traduction model specifically targeting vocals.

**Model & Dataset:** model is YourMT3+. 4 evaluation datasets. Training dataset isn't specified.

**Optimization & System:** conditioning, fine-tuning (on what?), constrained inference, quantization. Additional inputs: tempo & timed captions

**Feedback:** Please refrain from directly sharing your chatbot's output. It's not clear from your presentation why existing baseline fail on this task and how your method is going to alleviate that. Make sure you understand well the project you propose, as it covers technique that weren't covered in the course.

---

## Grading Rubrics

| Criteria | 9–10 (Excellent) | 7–8 (Good) | 4–6 (Satisfactory) | 1–3 (Needs Improvement) | Score |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **Novelty of GAI Application** | Highly original concept; tackles a compelling problem with innovative open-model utilization. | Clear, practical application with distinct original features or tailored workflows. | Conventional use case; limited original thought beyond standard applications. | Trivial wrapper around a model with minimal creativity or added value. | **/10** |
| **Difficulty of Adaptation** | Implements a novel model adaptation and inference-time improvement, supported by solid performance gains. | Combines model adaptation with explicit inference-time optimization (e.g., quantization, pruning, caching). | Adapts an existing model (e.g., fine-tuning, RAG, custom agent pipelines) beyond default usage. | Uses an existing model out-of-the-box without meaningful adaptation (insufficient). | **/10** |
| **Extensiveness of Evaluation** | Evaluates **4+ datasets** or distinct model variants with comprehensive metrics and benchmark analysis. | Evaluates **3 datasets** or distinct model variants with thorough comparative metrics. | Evaluates **2 datasets** or distinct model variants using basic validation metrics. | Evaluates **1 or 0 datasets**; superficial testing with minimal empirical validation. | **/10** |
| **Report & Code Repository** | Professional report paired with a well-structured, reproducible code repository and clear README setup. | Clear report and organized repository; code runs with minor setup effort. | Complete report, but repository lacks clear documentation or setup instructions. | Incomplete report or broken, unstructured repository without instructions. | **/10** |
