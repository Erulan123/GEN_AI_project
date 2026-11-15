"""
Scores a whole test set: every prediction is matched with its reference by name,
then scored with score.py and count_errors.py.

INPUTS:
    --references   a folder of reference files (e.g. CSD csv/ or mid/), or one JSON file
                   with several songs (e.g. N20EMv2 annotations.json)
    --predictions  a folder with the model's files. Each one must have the same name as
                   its reference (en001a.csv <-> en001a.mid), or as the song id in the JSON
OUTPUT:
    A table with one row per song (scores and error counts) and the average, printed
    and saved as CSV with --out.

Examples:
    python score_folder.py --references CSD/english/csv --predictions outputs/csd_en --out csd_en.csv
    python score_folder.py --references N20EMv2/annotations.json --split test --predictions outputs/n20em --out n20em.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from count_errors import ERROR_TYPES, count_errors
from score import list_json_songs, load_notes, read_json, clean_notes, score, select_format

READABLE = {".mid", ".midi", ".csv", ".json"}


def find_pairs(references, predictions, split=None):
    """Return (name, reference loader, prediction path) for every song found on both sides."""
    predicted = {p.stem: p for p in Path(predictions).iterdir() if p.suffix.lower() in READABLE}

    if Path(references).is_file():  # one JSON file with several songs
        songs = list_json_songs(references, split)
        refs = {song: (lambda song=song: clean_notes(read_json(references, song))) for song in songs}
    else:
        refs = {p.stem: (lambda p=p: load_notes(p))
                for p in sorted(Path(references).iterdir()) if p.suffix.lower() in READABLE}

    missing = sorted(set(refs) - set(predicted))
    if missing:
        print(f"Warning: no prediction for {len(missing)} reference(s), e.g. {missing[:5]}. Skipped.")
    return [(name, refs[name], predicted[name]) for name in sorted(refs) if name in predicted]


def main():
    parser = argparse.ArgumentParser(description="Score a whole test set.")
    parser.add_argument("--references", required=True, help="folder of references, or one JSON file")
    parser.add_argument("--predictions", required=True, help="folder of model outputs")
    parser.add_argument("--split", help="JSON only: keep only this split, e.g. test")
    parser.add_argument("--prediction-program", type=int,
                        help="MIDI only: instrument number of the singing track in the predictions")
    parser.add_argument("--out", help="save the table to this CSV file")
    args = parser.parse_args()

    rows = []
    for name, load_reference, prediction_path in find_pairs(args.references, args.predictions, args.split):
        reference = load_reference()
        prediction = load_notes(prediction_path, select_format(prediction_path), args.prediction_program)

        row = {"song": name, "reference_notes": len(reference), "predicted_notes": len(prediction)}
        for metric, (precision, recall, f1) in score(reference, prediction).items():
            row.update({f"{metric}_P": precision, f"{metric}_R": recall, f"{metric}_F1": f1})
        row.update(count_errors(reference, prediction))
        rows.append(row)

    if not rows:
        print("No matching reference/prediction pairs found.")
        return

    # Average: scores are averaged over songs (each song counts the same);
    # note and error counts are summed over the whole test set.
    columns = list(rows[0])
    total = {"song": "AVERAGE / TOTAL"}
    for column in columns[1:]:
        values = [row[column] for row in rows]
        is_count = column in ERROR_TYPES or column.endswith("_notes")
        total[column] = int(np.sum(values)) if is_count else float(np.mean(values))

    print(f"\n{len(rows)} songs scored.")
    for metric in ("COn", "COnP", "COnPOff"):
        print(f"{metric:8} F1 = {total[f'{metric}_F1']:.3f}")
    print("Errors:", ", ".join(f"{name} {total[name]}" for name in ERROR_TYPES))

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows + [total])
        print(f"Table saved to {args.out}")


if __name__ == "__main__":
    main()
