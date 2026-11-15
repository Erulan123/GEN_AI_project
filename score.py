"""
Scores one transcription: compares the notes the model wrote with the notes a
human wrote for the same recording, and prints COn, COnP and COnPOff.

INPUTS (two files, both read by this script):
    --reference   the notes written by hand (from the dataset)
    --prediction  the notes written by the model (e.g. the .mid from YourMT3+)
    Each one can be MIDI (.mid), CSV (.csv) or JSON (.json).

OUTPUT:
    The scores, printed in the terminal. No file is written.

Examples:
    python score.py --reference en001a.csv --prediction en001a.mid
    python score.py --reference annotations.json --reference-song 12 --prediction 12.mid
    python score.py --reference en001a.mid --prediction en001a.mid --prediction-program 65 --show-notes
"""

import argparse
import csv
import json
from pathlib import Path

import mir_eval
import numpy as np
import pretty_midi


# ----------------------------------------------------------------------------
# 1. FORMAT SELECTOR: which kind of file did we get as input?
# ----------------------------------------------------------------------------

SUPPORTED_FORMATS = ["midi", "csv", "json"]


def select_format(path, forced_format=None):
    """Return "midi", "csv" or "json". Uses forced_format if given, else the file extension."""
    if forced_format:
        if forced_format not in SUPPORTED_FORMATS:
            raise ValueError(f"Unknown format '{forced_format}'. Use one of {SUPPORTED_FORMATS}.")
        return forced_format

    extension = Path(path).suffix.lower()
    if extension in (".mid", ".midi"):
        return "midi"
    if extension == ".csv":
        return "csv"
    if extension == ".json":
        return "json"
    raise ValueError(f"Can't guess the format of '{path}'. Use --reference-format / --prediction-format.")


def load_notes(path, file_format=None, program=None, song_id=None):
    """Read an input file of any supported format and return it in the common note format."""
    chosen_format = select_format(path, file_format)
    if chosen_format == "midi":
        notes = read_midi(path, program)
    elif chosen_format == "csv":
        notes = read_csv(path)
    else:
        notes = read_json(path, song_id)
    return clean_notes(notes)


# ----------------------------------------------------------------------------
# 2. CONVERSION of the input file to the common note format:
#    one row per note, [onset_s, offset_s, midi_pitch]
# ----------------------------------------------------------------------------

# Instrument numbers that can hold the singing in a YourMT3+ MIDI file.
# Inside the model, singing is program 100, but its MIDI writer saves it as
# program 65 by default. Check on a real output file and pass --prediction-program if needed.
SINGING_PROGRAMS = [65, 100]


def read_midi(path, program=None):
    """MIDI file -> notes of one instrument (the singing)."""
    midi = pretty_midi.PrettyMIDI(str(path))
    instruments = [inst for inst in midi.instruments if not inst.is_drum]
    if not instruments:
        raise ValueError(f"No (non-drum) notes found in '{path}'.")

    if program is not None:
        chosen = [inst for inst in instruments if inst.program == program]
    elif len(instruments) == 1:
        chosen = instruments  # e.g. CSD: one track only
    else:
        chosen = [inst for inst in instruments if inst.program in SINGING_PROGRAMS]

    if not chosen:
        found = sorted({inst.program for inst in instruments})
        raise ValueError(
            f"Can't tell which track is the singing in '{path}'. Instrument numbers in the file: "
            f"{found}. Pass one with --reference-program / --prediction-program."
        )

    return [[note.start, note.end, note.pitch] for inst in chosen for note in inst.notes]


# Column names we accept, in order of preference.
# CSD files use "start", "end", "pitch". Our own format uses "onset_s", "offset_s", "midi_pitch".
CSV_COLUMNS = {
    "onset": ["onset_s", "start", "onset"],
    "offset": ["offset_s", "end", "offset"],
    "pitch": ["midi_pitch", "pitch", "note"],
}


def read_csv(path):
    """CSV file with a header row -> notes."""
    with open(path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return []

    columns = {}
    for role, candidates in CSV_COLUMNS.items():
        match = next((name for name in candidates if name in rows[0]), None)
        if match is None:
            raise ValueError(f"No {role} column in '{path}'. Columns found: {list(rows[0])}.")
        columns[role] = match

    return [
        [float(row[columns["onset"]]), float(row[columns["offset"]]), float(row[columns["pitch"]])]
        for row in rows
    ]


def read_json(path, song_id=None):
    """JSON file -> notes of one song.

    N20EMv2 annotations.json (checked on the real file):
        {"<song id>": {"split": "train" | "test" | ..., "midi": [[onset, offset, pitch], ...]}, ...}
    The MIR-ST500 style {"<song id>": [[onset, offset, pitch], ...]} and a plain list
    of [onset, offset, pitch] (one song per file) also work.
    """
    with open(path, encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict):
        if song_id is None:
            some_ids = list(data)[:5]
            raise ValueError(f"'{path}' holds several songs. Pass --reference-song, e.g. one of {some_ids}.")
        if str(song_id) not in data:
            raise ValueError(f"Song '{song_id}' not found in '{path}'.")
        data = data[str(song_id)]
        if isinstance(data, dict):
            data = data["midi"]

    return [[float(n[0]), float(n[1]), float(n[2])] for n in data]


def list_json_songs(path, split=None):
    """Song ids in a JSON file with several songs, optionally only one split (e.g. "test")."""
    with open(path, encoding="utf-8") as file:
        data = json.load(file)
    return [song for song, entry in data.items()
            if split is None or (isinstance(entry, dict) and entry.get("split") == split)]


def clean_notes(notes):
    """Remove notes with no length, sort by onset, return an (n, 3) array."""
    notes = np.array(notes, dtype=float).reshape(-1, 3)
    notes = notes[notes[:, 1] > notes[:, 0]]
    return notes[np.argsort(notes[:, 0], kind="stable")]


def save_notes(notes, path):
    """Write notes in the common format as CSV, e.g. to check a conversion by eye."""
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["onset_s", "offset_s", "midi_pitch"])
        writer.writerows(notes.tolist())


# ----------------------------------------------------------------------------
# 3. SCORING
# ----------------------------------------------------------------------------
#
# Definitions
#
#   Onset:   the time a note starts, in seconds.
#   Offset:  the time a note ends, in seconds.
#   Cents:   a pitch distance. 100 cents = 1 semitone, 1200 cents = 1 octave.
#
#   A predicted note is "correct" (a true positive, TP) when it matches a reference
#   note. Each reference note can be matched by at most one predicted note.
#   FP (false positive): a predicted note that matches nothing (extra or wrong note).
#   FN (false negative): a reference note that no predicted note matched (missed note).
#
#   Precision = TP / (TP + FP)   of the notes the model wrote, the share that are correct
#   Recall    = TP / (TP + FN)   of the reference notes, the share the model found
#   F1        = 2 * P * R / (P + R)   one score, high only when both P and R are high
#
#   COn, COnP and COnPOff each give a precision, a recall and an F1.
#   They only differ in what a predicted note must get right to count as correct,
#   and each one is stricter than the one before:
#
#   COn     (Correct Onset): the onset is within 50 ms of the reference onset.
#           Pitch and offset are ignored.
#   COnP    (Correct Onset and Pitch): COn, and the pitch is within 50 cents
#           (half a semitone) of the reference pitch. Our main score.
#   COnPOff (Correct Onset, Pitch and Offset): COnP, and the offset is within
#           max(50 ms, 20% of the reference note length) of the reference offset.
#           Long notes get more room for their end than short ones.

ONSET_TOLERANCE = 0.05   # seconds (50 ms)
PITCH_TOLERANCE = 50.0   # cents (half a semitone)
OFFSET_RATIO = 0.2       # offset may be off by 20% of the reference note length...
OFFSET_MIN = 0.05        # ...or by 50 ms, whichever is larger


def score(reference_notes, predicted_notes):
    """Return {"COn": (P, R, F1), "COnP": (P, R, F1), "COnPOff": (P, R, F1)}."""
    ref_intervals, pred_intervals = reference_notes[:, :2], predicted_notes[:, :2]
    ref_hz = mir_eval.util.midi_to_hz(reference_notes[:, 2])  # mir_eval wants pitches in Hz
    pred_hz = mir_eval.util.midi_to_hz(predicted_notes[:, 2])

    con = mir_eval.transcription.onset_precision_recall_f1(
        ref_intervals, pred_intervals, onset_tolerance=ONSET_TOLERANCE)

    # offset_ratio=None tells mir_eval to ignore the offsets
    conp = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_hz, pred_intervals, pred_hz,
        onset_tolerance=ONSET_TOLERANCE, pitch_tolerance=PITCH_TOLERANCE,
        offset_ratio=None)[:3]

    conpoff = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_hz, pred_intervals, pred_hz,
        onset_tolerance=ONSET_TOLERANCE, pitch_tolerance=PITCH_TOLERANCE,
        offset_ratio=OFFSET_RATIO, offset_min_tolerance=OFFSET_MIN)[:3]

    return {"COn": con, "COnP": conp, "COnPOff": conpoff}


def print_scores(scores):
    print(f"{'':10}{'Precision':>10}{'Recall':>10}{'F1':>10}")
    for name, (precision, recall, f1) in scores.items():
        print(f"{name:10}{precision:10.3f}{recall:10.3f}{f1:10.3f}")


# ----------------------------------------------------------------------------
# 4. COMMAND LINE
# ----------------------------------------------------------------------------

def add_input_arguments(parser):
    """The input options shared by score.py and count_errors.py."""
    parser.add_argument("--reference", required=True, help="input: notes written by hand (dataset)")
    parser.add_argument("--prediction", required=True, help="input: notes written by the model")
    for side in ("reference", "prediction"):
        parser.add_argument(f"--{side}-format", choices=SUPPORTED_FORMATS,
                            help="only needed if the file extension is unusual")
        parser.add_argument(f"--{side}-program", type=int,
                            help="MIDI only: instrument number of the singing track")
        parser.add_argument(f"--{side}-song", help="JSON only: song id, if the file holds several songs")
    parser.add_argument("--show-notes", action="store_true", help="print the first 5 notes of each input")


def load_inputs(args):
    """Read both input files and print how many notes each one has."""
    reference = load_notes(args.reference, args.reference_format, args.reference_program, args.reference_song)
    prediction = load_notes(args.prediction, args.prediction_format, args.prediction_program, args.prediction_song)
    print(f"Reference: {len(reference)} notes | Prediction: {len(prediction)} notes")

    if args.show_notes:
        for label, notes in (("Reference", reference), ("Prediction", prediction)):
            print(f"\n{label}, first notes [onset_s, offset_s, midi_pitch]:")
            print(np.round(notes[:5], 3))
        print()
    return reference, prediction


def main():
    parser = argparse.ArgumentParser(description="Score one transcription with COn, COnP and COnPOff.")
    add_input_arguments(parser)
    reference, prediction = load_inputs(parser.parse_args())
    print_scores(score(reference, prediction))


if __name__ == "__main__":
    main()
