#!/usr/bin/env python3
"""
bodmas_restore.py
=================
Restore the two PE header fields that BODMAS zeroes out to disarm its malware
binaries, so that the EMBER feature extractor can parse them correctly.

BODMAS distributes malware in *disarmed* form: the PE COFF header `Machine`
field and the Optional header `Subsystem` field are set to 0 to prevent
accidental execution. With `Machine = 0`, LIEF fails to parse the import
table, which corrupts ~39 of the 2,381 EMBER features (the import group,
including the highest-gain `imports_4` bucket). Restoring the two fields makes
extraction faithful again; the byte-histogram and byte-entropy groups are
unaffected either way.

This utility rewrites ONLY those two fields, in place on a *copy*, by patching
the raw bytes at the offsets given by the PE header — it does not change any
section data, so the PRISM section table and the EMBER byte/string features are
identical to the original.

Restored values used (matching the paper, Section VI-E):
    Machine   -> 0x014c   (IMAGE_FILE_MACHINE_I386)
    Subsystem -> 0x0002   (IMAGE_SUBSYSTEM_WINDOWS_GUI)

SECURITY NOTE
-------------
Restored binaries are functional malware. Do NOT distribute them and do NOT run
them outside an isolated analysis VM. This script is released for
reproducibility of the feature-extraction pipeline only. No binaries (disarmed
or restored) are included in the PRISM release.

Usage
-----
    # single file
    python bodmas_restore.py --in disarmed.bin --out restored.bin

    # batch a directory (writes <name>.restored next to each, or to --outdir)
    python bodmas_restore.py --indir /path/altered --outdir /path/restored
"""

import argparse
import struct
import sys
from pathlib import Path

MACHINE_I386 = 0x014C
SUBSYSTEM_GUI = 0x0002


def restore_pe_bytes(data: bytes,
                     machine: int = MACHINE_I386,
                     subsystem: int = SUBSYSTEM_GUI) -> bytes:
    """Return a copy of `data` with Machine and Subsystem restored.

    Parses the PE structure minimally from raw bytes:
      - DOS header `e_lfanew` at offset 0x3C points to the PE signature.
      - COFF File Header starts at e_lfanew + 4; `Machine` is its first 2 bytes.
      - Optional Header starts at e_lfanew + 24; its first 2 bytes are the Magic
        (0x10B = PE32, 0x20B = PE32+); `Subsystem` is at optional-header
        offset 68 for both PE32 and PE32+ (the layout is identical up to that
        field).
    Raises ValueError if the file is not a recognisable PE.
    """
    if len(data) < 0x40 or data[0:2] != b"MZ":
        raise ValueError("not a DOS/PE file (missing MZ magic)")

    e_lfanew = struct.unpack_from("<I", data, 0x3C)[0]
    if e_lfanew + 4 > len(data) or data[e_lfanew:e_lfanew + 4] != b"PE\x00\x00":
        raise ValueError("missing PE signature at e_lfanew")

    buf = bytearray(data)

    # COFF File Header: Machine = first WORD after the PE signature.
    coff = e_lfanew + 4
    struct.pack_into("<H", buf, coff, machine)

    # Optional Header begins 20 bytes after the start of the COFF header.
    opt = coff + 20
    if opt + 2 > len(buf):
        raise ValueError("file truncated before optional header")
    magic = struct.unpack_from("<H", buf, opt)[0]
    if magic not in (0x10B, 0x20B):
        raise ValueError(f"unexpected optional-header magic 0x{magic:04x}")

    # Subsystem is a WORD at optional-header offset 68 for both PE32 / PE32+.
    subsystem_off = opt + 68
    if subsystem_off + 2 > len(buf):
        raise ValueError("file truncated before Subsystem field")
    struct.pack_into("<H", buf, subsystem_off, subsystem)

    return bytes(buf)


def restore_file(in_path: Path, out_path: Path) -> bool:
    try:
        data = in_path.read_bytes()
        out_path.write_bytes(restore_pe_bytes(data))
        return True
    except Exception as e:  # noqa: BLE001 - report and continue in batch mode
        print(f"  FAIL {in_path.name}: {e}", file=sys.stderr)
        return False


def main():
    ap = argparse.ArgumentParser(description="Restore disarmed BODMAS PE headers.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--in", dest="infile", help="single input binary")
    g.add_argument("--indir", help="directory of input binaries (batch)")
    ap.add_argument("--out", help="output path (single-file mode)")
    ap.add_argument("--outdir", help="output directory (batch mode)")
    ap.add_argument("--machine", type=lambda x: int(x, 0), default=MACHINE_I386)
    ap.add_argument("--subsystem", type=lambda x: int(x, 0), default=SUBSYSTEM_GUI)
    args = ap.parse_args()

    if args.infile:
        out = Path(args.out) if args.out else Path(args.infile).with_suffix(".restored")
        ok = restore_file(Path(args.infile), out)
        print(f"{'OK' if ok else 'FAIL'}: {out}")
        sys.exit(0 if ok else 1)

    indir = Path(args.indir)
    outdir = Path(args.outdir) if args.outdir else indir
    outdir.mkdir(parents=True, exist_ok=True)
    files = [p for p in indir.iterdir() if p.is_file()]
    ok = fail = 0
    for p in files:
        dst = outdir / (p.name + ".restored" if outdir == indir else p.name)
        if restore_file(p, dst):
            ok += 1
        else:
            fail += 1
    print(f"Restored {ok}/{ok + fail} ({fail} failed) -> {outdir}")


if __name__ == "__main__":
    main()
