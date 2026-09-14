#!/usr/bin/env python3
"""normalize_nfc.py — optional Unicode NFC normalisation.

The released trajectory file is shipped exactly as it was used for training, so a
handful of records still carry decomposed (NFD) Vietnamese characters (data card 4.6).
Use this only if your pipeline requires NFC; the result will no longer be byte-identical
to the file the paper's models were trained on.

    python3 scripts/normalize_nfc.py IN.jsonl OUT.jsonl
"""
import json
import sys
import unicodedata


def walk(x):
    if isinstance(x, str):
        return unicodedata.normalize("NFC", x)
    if isinstance(x, list):
        return [walk(v) for v in x]
    if isinstance(x, dict):
        return {walk(k): walk(v) for k, v in x.items()}
    return x


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    src, dst = sys.argv[1], sys.argv[2]
    changed = 0
    with open(src, encoding="utf-8") as fi, open(dst, "w", encoding="utf-8") as fo:
        for line in fi:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out = walk(row)
            if out != row:
                changed += 1
            fo.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"normalised {changed} record(s) -> {dst}")
    return 0


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2 or sys.argv[1] in ("--help", "-h"):
        print(__doc__)
        print("Usage: python3 scripts/normalize_nfc.py <file.jsonl> [out.jsonl]")
        raise SystemExit(0)
    sys.exit(main())
