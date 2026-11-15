"""
Counts the types of errors in one transcription, to explain the scores of score.py.

INPUTS: the same two files as score.py (--reference and --prediction, MIDI, CSV or JSON).
OUTPUT: the error counts, printed in the terminal.

Example:
    python count_errors.py --reference en001a.csv --prediction en001a.mid
"""

import argparse

import numpy as np

from score import add_input_arguments, load_inputs

# ----------------------------------------------------------------------------
# Definitions
#
#   To find which predicted note belongs to which reference note, we look at how
#   much they overlap in time. Each predicted note belongs to the reference note
#   it overlaps the most.
#
#   Per reference note (each one gets at most one of the first three labels):
#     merged        one predicted note covers more than half of this note AND of
#                   at least one other reference note: two sung notes became one
#     missed        no predicted note belongs to it
#     split         two or more predicted notes belong to it: one sung note was cut in several
#   If exactly one predicted note belongs to it, we compare the two:
#     octave_error  the pitch is off by a whole number of octaves (12, 24... semitones)
#     wrong_pitch   the pitch is off by half a semitone or more, but not by whole octaves
#     onset_off     the onset is more than 50 ms away (can happen together with a pitch error)
#     correct       none of the above
#
#   Per predicted note:
#     extra         it overlaps no reference note at all (e.g. a breath taken as a note)
# ----------------------------------------------------------------------------

ONSET_TOLERANCE = 0.05   # seconds, same as in score.py
PITCH_TOLERANCE = 0.5    # semitones (50 cents), same as in score.py
COVER_SHARE = 0.5        # "covers" a reference note = overlaps more than half of its length

ERROR_TYPES = ["correct", "wrong_pitch", "octave_error", "onset_off",
               "split", "merged", "missed", "extra"]


def overlap(a, b):
    """Time two notes share, in seconds."""
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def count_errors(reference_notes, predicted_notes):
    """Return a dict with the number of notes per error type."""
    counts = dict.fromkeys(ERROR_TYPES, 0)
    belongs_to = {i: [] for i in range(len(reference_notes))}
    merged_refs = set()

    for pred in predicted_notes:
        overlaps = [overlap(pred, ref) for ref in reference_notes]
        if not overlaps or max(overlaps) == 0:
            counts["extra"] += 1
            continue
        belongs_to[int(np.argmax(overlaps))].append(pred)

        covered = [i for i, ref in enumerate(reference_notes)
                   if overlaps[i] > COVER_SHARE * (ref[1] - ref[0])]
        if len(covered) >= 2:
            merged_refs.update(covered)

    for i, ref in enumerate(reference_notes):
        preds = belongs_to[i]
        if i in merged_refs:
            counts["merged"] += 1
        elif not preds:
            counts["missed"] += 1
        elif len(preds) >= 2:
            counts["split"] += 1
        else:
            pred = preds[0]
            semitones = pred[2] - ref[2]
            octaves = round(semitones / 12)
            pitch_wrong = abs(semitones) >= PITCH_TOLERANCE
            onset_wrong = abs(pred[0] - ref[0]) > ONSET_TOLERANCE

            if pitch_wrong and octaves != 0 and abs(semitones - 12 * octaves) < PITCH_TOLERANCE:
                counts["octave_error"] += 1
            elif pitch_wrong:
                counts["wrong_pitch"] += 1
            if onset_wrong:
                counts["onset_off"] += 1
            if not pitch_wrong and not onset_wrong:
                counts["correct"] += 1

    return counts


def print_errors(counts, n_reference):
    print(f"{'Error type':15}{'Notes':>7}{'% of reference':>17}")
    for name in ERROR_TYPES:
        share = 100 * counts[name] / n_reference if n_reference else 0
        print(f"{name:15}{counts[name]:7d}{share:16.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Count the error types in one transcription.")
    add_input_arguments(parser)
    reference, prediction = load_inputs(parser.parse_args())
    print_errors(count_errors(reference, prediction), len(reference))


if __name__ == "__main__":
    main()
