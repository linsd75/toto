#!/usr/bin/env python3
import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox


def convert_ndjson_to_json(input_path: Path) -> Path:
    output_path = input_path.with_suffix(".json")
    records = []

    with input_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{input_path.name} line {line_no}: {e}") from e

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    return output_path


def select_and_convert():
    file_paths = filedialog.askopenfilenames(
        title="Select NDJSON files",
        filetypes=[
            ("NDJSON files", "*.ndjson *.jsonl"),
            ("All files", "*.*"),
        ],
    )

    if not file_paths:
        return

    ok = []
    failed = []

    for file_path in file_paths:
        path = Path(file_path)
        try:
            out = convert_ndjson_to_json(path)
            ok.append(f"{path.name} -> {out.name}")
        except Exception as e:
            failed.append(f"{path.name}: {e}")

    msg = []
    if ok:
        msg.append("Converted:\n" + "\n".join(ok))
    if failed:
        msg.append("\nFailed:\n" + "\n".join(failed))

    if failed:
        messagebox.showwarning("Done with errors", "\n".join(msg))
    else:
        messagebox.showinfo("Done", "\n".join(msg))


def main():
    root = tk.Tk()
    root.title("NDJSON to JSON Converter")
    root.geometry("420x180")
    root.resizable(False, False)

    frame = tk.Frame(root, padx=16, pady=16)
    frame.pack(fill="both", expand=True)

    tk.Label(
        frame,
        text="Select one or more NDJSON files.\nEach output .json is saved in the same folder.",
        justify="center",
    ).pack(pady=(0, 14))

    tk.Button(
        frame,
        text="Select Files and Convert",
        command=select_and_convert,
        width=28,
        height=2,
    ).pack()

    root.mainloop()


if __name__ == "__main__":
    main()
