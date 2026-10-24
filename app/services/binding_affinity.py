"""
Complementariedad y energía de dúplex miRNA–mRNA.

Biopython: complemento reverso y alfabeto de secuencia.
NumPy: comparación base a base y suma de stacks.
La energía libre (ΔG, kcal/mol, 37 °C) usa los parámetros de vecinos
más próximos de dúplex ARN (Xia et al., 1998), con penalización por
mismatch. No sustituye a ViennaRNA; es una estimación reproducible
para el dataset de aprendizaje.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from Bio.Seq import Seq

# Iniciación de dúplex y penalización de extremo A-U (kcal/mol)
DUPLEX_INIT = 4.09
TERMINAL_AU = 0.45
MISMATCH_PENALTY = 1.25
WOBBLE_STACK = -0.55

# Stack 5'->3' de la hebra diana cuando el par es Watson-Crick (kcal/mol)
WC_STACK = {
    "AA": -0.93,
    "AU": -1.10,
    "UA": -1.33,
    "UU": -0.93,
    "CA": -2.11,
    "CU": -2.08,
    "GA": -2.35,
    "GU": -2.24,
    "AC": -2.24,
    "AG": -2.08,
    "UC": -2.35,
    "UG": -2.11,
    "CG": -2.36,
    "GC": -3.26,
    "GG": -3.26,
    "CC": -3.26,
}

_COMP = {"A": "T", "T": "A", "G": "C", "C": "G"}


def _norm(seq: str) -> str:
    return (seq or "").upper().replace("U", "T")


def reverse_complement(seq: str) -> str:
    return str(Seq(_norm(seq)).reverse_complement())


def _pair_kind(top: str, bottom: str) -> str:
    """top = diana 5'->3'; bottom = base del miRNA que enfrenta (ya en la misma columna)."""
    if top not in "ACGT" or bottom not in "ACGT":
        return "mismatch"
    if _COMP.get(top) == bottom:
        return "wc"
    if {top, bottom} == {"G", "T"}:
        return "wobble"
    return "mismatch"


def _site_window(mirna: str, mrna: str, seed_pos: Optional[int]) -> tuple[str, int]:
    """
    Ventana del mRNA alineada al complemento reverso del miRNA completo.
    seed_pos es el inicio del RC del seed (nt 2–8) dentro del mRNA.
    """
    mirna = _norm(mirna)
    mrna = _norm(mrna)
    length = len(mirna)
    if length == 0 or not mrna:
        return "", 0

    if seed_pos is None:
        start = 0
    else:
        # RC(mirna[1:8]) ocupa los índices [L-8, L-1) del RC completo
        seed_offset_in_rc = max(length - 8, 0)
        start = int(seed_pos) - seed_offset_in_rc

    if start < 0:
        start = 0
    window = mrna[start : start + length]
    return window, start


def _au_accessibility(mrna: str, start: int, length: int, flank: int = 15) -> float:
    mrna = _norm(mrna)
    lo = max(0, start - flank)
    hi = min(len(mrna), start + length + flank)
    window = mrna[lo:hi]
    if not window:
        return 0.0
    bases = np.frombuffer(window.encode("ascii"), dtype=np.uint8)
    au = np.isin(bases, np.frombuffer(b"AT", dtype=np.uint8)).mean()
    return round(float(au), 4)


def score_binding(mirna: str, mrna: str, seed_pos: Optional[int] = None) -> Dict[str, float]:
    """
    Alinea el miRNA (antiparalelo) contra una ventana del mRNA y devuelve
    features biológicas para el modelo.
    """
    mirna_n = _norm(mirna)
    rc = reverse_complement(mirna_n)
    window, start = _site_window(mirna_n, mrna, seed_pos)
    n = min(len(window), len(rc), len(mirna_n))
    if n == 0:
        return _empty()

    target = np.array(list(window[:n]))
    guide_rc = np.array(list(rc[:n]))
    # La base del miRNA que enfrenta target[i] es mirna[n-1-i] solo si window == rc.
    # Comparamos target contra el RC esperado: cada columna es un par del dúplex.
    kinds = [_pair_kind(target[i], guide_rc[i]) for i in range(n)]
    kind_arr = np.array(kinds)
    n_wc = int(np.sum(kind_arr == "wc"))
    n_wobble = int(np.sum(kind_arr == "wobble"))
    n_mismatch = int(np.sum(kind_arr == "mismatch"))

    stack_sum = 0.0
    for i in range(n - 1):
        if kinds[i] == "mismatch" or kinds[i + 1] == "mismatch":
            stack_sum += MISMATCH_PENALTY
            continue
        if kinds[i] == "wobble" or kinds[i + 1] == "wobble":
            stack_sum += WOBBLE_STACK
            continue
        dinuc = window[i : i + 2]
        stack_sum += WC_STACK.get(dinuc, -1.0)

    terminal = 0.0
    if n and window[0] in "AT":
        terminal += TERMINAL_AU
    if n and window[n - 1] in "AT":
        terminal += TERMINAL_AU

    delta_g = round(float(DUPLEX_INIT + stack_sum + terminal), 3)
    complementarity = round(float((n_wc + 0.5 * n_wobble) / n), 4)
    seed = mirna_n[1:8] if len(mirna_n) >= 8 else mirna_n
    seed_rc = reverse_complement(seed)
    seed_match = 1 if seed_pos is not None and seed_rc in _norm(mrna) else 0

    # Score de alineamiento acotado: más complementariedad y ΔG más negativo suben el score
    align = round(float(np.clip(complementarity * 0.65 + min(abs(min(delta_g, 0.0)) / 40.0, 1.0) * 0.35, 0, 1)), 4)

    return {
        "seed_match": seed_match,
        "seed_position": int(seed_pos) if seed_pos is not None else None,
        "match_length": int(n_wc + n_wobble),
        "num_matches": n_wc,
        "mismatches": n_mismatch,
        "gaps": 0,
        "indels": n_wobble,
        "complementarity": complementarity,
        "alignment_score": align,
        "site_position": int(start),
        "site_length": int(n),
        "delta_g": delta_g,
        "accessibility": _au_accessibility(_norm(mrna), start, n),
    }


def _empty() -> Dict[str, float]:
    return {
        "seed_match": 0,
        "seed_position": None,
        "match_length": 0,
        "num_matches": 0,
        "mismatches": 0,
        "gaps": 0,
        "indels": 0,
        "complementarity": 0.0,
        "alignment_score": 0.0,
        "site_position": None,
        "site_length": 0,
        "delta_g": 0.0,
        "accessibility": 0.0,
    }


def describe_binding_frame(df) -> dict:
    """Resumen estadístico con Pandas/NumPy para el informe."""
    import pandas as pd

    if df is None or len(df) == 0:
        return {}
    cols = [c for c in ("complementarity", "delta_g", "accessibility", "mismatches", "alignment_score", "ml_probability") if c in df.columns]
    desc = df[cols].apply(pd.to_numeric, errors="coerce").describe().round(4)
    return {
        "n": int(len(df)),
        "mean_complementarity": float(np.nanmean(pd.to_numeric(df["complementarity"], errors="coerce"))) if "complementarity" in df.columns else None,
        "mean_delta_g": float(np.nanmean(pd.to_numeric(df["delta_g"], errors="coerce"))) if "delta_g" in df.columns else None,
        "seed_match_fraction": float(np.nanmean(pd.to_numeric(df["seed_match"], errors="coerce"))) if "seed_match" in df.columns else None,
        "describe": desc.to_dict(),
    }
