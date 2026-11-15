# MSGAI26 Project: Fine-tuning YourMT3+ for vocal to MIDI transcription

Group 8 · Elise Høimyr, Jue Wang, Erulan Ibraimov · Nov 14, 2026

## 1. Summary

Most of us have noticed that voice-to-text models are getting better and better. Voice-to-MIDI models are a comparable tool, but for musicians: you sing a melody and you get the notes. However, they are a lot further behind. Even the best ones consistently make errors that have to be fixed manually.

In this project we take YourMT3+, which is an open model that already turns music into notes, and we fine-tune it with LoRA on singing recordings it has never seen. Then we make it faster and lighter, and we measure how much quality we lose each time.

- **Baseline:** YourMT3+ as it is released, without any change.
- **Adaptation:** LoRA fine-tuning on singing recordings where the notes were written by hand.
- **Inference optimization:** quantization, batching and a smaller version of the model.
- **Evaluation:** 4 test sets and 7 versions of the model. We measure the note accuracy, the latency and the memory.

We removed the following from the original proposal: syllables from Whisper, tempo grid, vocal separation.&#32;

## 2. Problem and users

First, here are some definitions:

- **MIDI file:** it's basically a digital score. It maps out the notes by timestamps, and it can be edited in music production software.
- **DAW:** the software music producers use to create a song, for example Ableton Live.
- **Pitch:** the note that is sung: do, re, mi, or A, B, C, etc.
- **Onset and offset:** the moments where a note starts and stops.
- **Tuner:** a tool that tells you which note is being played or sung right now. Guitarists use it to tune their guitar.

**Who it is for:** Music producers and songwriters who want to simply sing or hum the notes they have in mind and have them directly in their DAW, instead of writing the score manually. It is also useful for people who build singing datasets and need the notes.

**The task:**

|  |  |
| --- | --- |
| Input | A recording of one singing voice, without instruments |
| Output | A MIDI file with the pitch, the onset and the offset of each sung note |

**The problem:** Even the converter that runs in Ableton Live, which is one of the most highly ranked, still makes errors consistently. In Elise's demo, she records a sung line, and one note comes out a semitone off and the timing is not accurate. So the producer has to fix it by hand.

**What we want:**

- The right pitch, without octave errors.
- One sung note gives one MIDI note. Vibrato or a slide should not cut it into several notes.
- Onsets within 50 ms of the reference, which is the standard tolerance.

**Out of scope: **Full songs with instruments, several voices and lyrics. We only look at the voice alone, so we don't need vocal separation.

## 3. Model

We use [YourMT3+](https://arxiv.org/abs/2407.04822) (Chang et al., MLSP 2024). It has an encoder that reads the audio, in pieces of about 2 seconds, and a decoder that writes the notes one token at a time, the same way a language model writes words. So it is a generative model, and what we learned in the course applies to it directly.

- It is small: about 46M parameters for the largest version, so it fits easily on our GPU (RTX 5080, 16 GB).
- The [training code](https://huggingface.co/spaces/mimbres/YourMT3/tree/main/amt/src) and the [checkpoints](https://huggingface.co/mimbres/YourMT3) are public.
- It can transcribe singing directly, which most open transcription models can't.

**Tuners work well, so why would a transcription model fail?** A tuner only has to say which pitch it hears right now. A transcription model also has to decide where each note starts and stops, and that is where it goes wrong. When we sing, the pitch moves inside a note (vibrato, slides), so the model can cut one note into several, or merge two notes into one.

**Why YourMT3+ in particular:**

1. **It has seen very little singing.** It was trained on 10 datasets, and only 2 of them contain singing (MIR-ST500 and CMedia). Both are pop songs with instruments. A voice alone, other languages, children or amateur singers are new to it.
2. **Its note starts are not precise.** In the paper, the singing score on MIR-ST500 is 70.6 (onset F1) when a note may be 50 ms off, but 81.1 when it may be 100 ms off. So a lot of notes are found, but placed 50 to 100 ms too early or too late. The authors say this themselves.

**How fine-tuning helps:** We show the model more singing, from different singers and languages, where the note starts were written by hand.

## 4. Data

We only keep datasets where the notes are matched to the audio, that we can download for free or with a quick application, and that were **not** used to train YourMT3+.

| Dataset | Content | Licence and access | Use |
| --- | --- | --- | --- |
| [N20EMv2](https://zenodo.org/records/10814703) | English songs by amateur singers, about 8 h, notes labelled by two music experts | CC BY-SA 4.0, open download (11.5 GB) | Training and test set 1 |
| [CSD](https://zenodo.org/record/4916302) | 50 English and 50 Korean children's songs, one professional singer, 2 keys each (200 recordings), MIDI made by hand | CC BY-NC-SA 4.0, open download (1.9 GB) | Training and test sets 2 and 3 |
| [ISMIR2014](https://riuma.uma.es/entities/publication/5256d1cf-45a9-42c9-bfae-3527d46a3cdb) (Molina et al.) | 38 short recordings, 19 min, children and adults | CC BY-NC-ND 4.0, free download | Test set 4 only |
| [GTSinger](https://huggingface.co/datasets/AaronZ345/GTSinger) | 80 h, 9 languages, 20 professional singers, note pitch and duration | CC BY-NC-SA 4.0, open on Hugging Face | Extra, only after we check the note timing by ear |
| M4Singer, Opencpop | Mandarin pop, 30 h and 5 h | Non-commercial, application form | We apply now and use them only if the access comes in time |
| TONAS | 72 flamenco recordings, 36 min | Free on request, internal use only | Extra test if the access comes |

**Training set.** N20EMv2 training singers + CSD English songs 1 to 40 (both keys).

**Test sets.**

1. N20EMv2, singers we kept out of training.
2. CSD English, songs 41 to 50: same singer, new songs.
3. CSD Korean, all 50 songs: a language the fine-tuning never saw.
4. ISMIR2014: a dataset we never train on, with children's voices.

All licences allow coursework. We don't put the data in the repo, a script downloads it.

**To confirm at download:** the exact number of recordings in N20EMv2, and whether it comes with an official split.

## 5. Method and hypotheses

We have two hypotheses, one for quality and one for speed.

**H1: fine-tuning with LoRA makes the notes more accurate** (fine-tuning from Lecture 4, Transformer from Lecture 5, LoRA from Lecture 8).

- We freeze YourMT3+ and add small LoRA matrices to its attention layers. We only train those, with the usual next-token cross-entropy on the note tokens.
- **What we expect:** the note score goes up on all 4 test sets compared to the released model, and most on the ones that are far from pop songs (children's songs, Korean, amateur singers).
- We compare two LoRA ranks (4 and 16) and one full fine-tune. For each we look at the score, the number of trained parameters and the training time.

**H2: the fine-tuned model can run cheaper with almost no loss** (inference optimization; quantization and batching are named in the project description).

- **Goal:** find the fastest setting that stays within 1 F1 point of our best model.
- What we change:
  - **Precision:** the weights are stored in 32 bits (FP32). We try 16 bits (FP16) and 8 bits (INT8).
  - **Batch size:** the audio is cut in pieces of 2 seconds, so we can transcribe many pieces at once.
  - **Model size:** YourMT3+ is released in several sizes (about 30M to 46M parameters). We test if a smaller one is good enough.
  - **LoRA merged or not:** if we merge the LoRA matrices into the weights, they should not cost anything extra at inference.
- **What we expect:** FP16 and batching make it clearly faster for free. INT8 saves memory but may cost accuracy.
- **Result:** one plot of quality against latency, with one point per setting.

We keep it to these two on purpose, because the project is short and both come from the course.

## 6. Evaluation

**Quality.** We compare the notes of the model with the notes a human wrote for the same recording, with the standard `mir_eval` library. A note of the model counts as correct when:

- **COn:** it starts within 50 ms of the reference note.
- **COnP:** it starts within 50 ms and the pitch is within half a semitone. This is our main score.
- **COnPOff:** same as COnP, and it also stops close to the reference (50 ms or 20% of the note length).

For each one we report the F1 score. We also count the types of errors from section 3.

**System metrics.** Latency per minute of audio, peak GPU memory, throughput (minutes of audio per second), number of trained parameters.

**Results matrix.** Every cell gets filled: 4 test sets and 7 variants.

| Variant | Test 1 N20EMv2 | Test 2 CSD-EN | Test 3 CSD-KO | Test 4 ISMIR2014 | Latency | Memory |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline: YourMT3+ as released |  |  |  |  |  |  |
| Baseline: smaller YourMT3+ checkpoint |  |  |  |  |  |  |
| H1: LoRA rank 4 |  |  |  |  |  |  |
| H1: LoRA rank 16 |  |  |  |  |  |  |
| H1: full fine-tune |  |  |  |  |  |  |
| H2: best model + FP16 |  |  |  |  |  |  |
| H2: best model + INT8 |  |  |  |  |  |  |

## 7. Risks and fallbacks

| Risk | Fallback |
| --- | --- |
| The YourMT3+ training code is custom (PyTorch Lightning, own data format), so fine-tuning may be harder to set up than with Hugging Face models | Before 17 Nov we check that we can overfit 20 clips. If LoRA is the problem, we train only the decoder and freeze the encoder |
| Fine-tuning YourMT3+ still doesn't work on 24 Nov (go/no-go) | We switch the base model to Whisper, as in our draft v2. Data, test sets and scoring stay the same |
| The released model is already good on our test sets | We report it honestly, focus on the hardest test set and on the efficiency study |
| The datasets have different label formats (MIDI, CSV, JSON) | One converter script to a common format: onset, offset, pitch |
| INT8 breaks the model | We report FP32 and FP16 only |
| Not enough GPU time | The model is small (46M) and we only train LoRA; we start with CSD alone |

8. ## Timeline

The weekly tasks per person are in the document "Weekly to-do".

## References

- Chang, Benetos, Kirchhoff, Dixon. *YourMT3+: Multi-instrument Music Transcription with Enhanced Transformer Architectures and Cross-dataset Stem Augmentation.* MLSP 2024. [arXiv:2407.04822](https://arxiv.org/abs/2407.04822) · [code](https://github.com/mimbres/YourMT3)
- Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
- Gu et al. *Automatic Lyric Transcription and Automatic Music Transcription from Multimodal Singing* (N20EMv2). ACM TOMM 2024. [Zenodo](https://zenodo.org/records/10814703)
- Choi et al. *Children's Song Dataset for Singing Voice Research.* ISMIR 2020 LBD. [Zenodo](https://zenodo.org/record/4916302)
- Molina et al. *Evaluation Framework for Automatic Singing Transcription.* ISMIR 2014. [Dataset](https://riuma.uma.es/entities/publication/5256d1cf-45a9-42c9-bfae-3527d46a3cdb)
- Zhang et al. *GTSinger.* NeurIPS 2024. [arXiv:2409.13832](https://arxiv.org/abs/2409.13832)
- Raffel et al. *mir\_eval.* ISMIR 2014. [GitHub](https://github.com/craffel/mir_eval)
