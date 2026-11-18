"""Minimal subjective-logic toolkit for the CPAG (Josang 2016 notation).
Opinion = (b, d, u, a) with b + d + u = 1."""
import numpy as np

W = 2.0  # non-informative prior weight


def op(b, d, u, a=0.5):
    assert abs(b + d + u - 1) < 1e-9
    return np.array([b, d, u, a], dtype=float)


def vacuous(a):
    return op(0.0, 0.0, 1.0, a)


def P(w):
    """Projected probability."""
    return w[0] + w[3] * w[2]


def discount(w_trust, w_claim):
    """R1: trust transitivity (Josang probability-sensitive form is avoided; classic belief discount)."""
    b = w_trust[0] * w_claim[0]
    d = w_trust[0] * w_claim[1]
    u = 1 - b - d
    return op(b, d, u, w_claim[3])


def conj(w1, w2):
    """R2: binomial multiplication (Josang & McAnally 2004)."""
    b1, d1, u1, a1 = w1
    b2, d2, u2, a2 = w2
    a = a1 * a2
    d = d1 + d2 - d1 * d2
    denom = 1 - a
    if denom < 1e-12:
        b = b1 * b2
    else:
        b = b1 * b2 + ((1 - a1) * a2 * b1 * u2 + a1 * (1 - a2) * u1 * b2) / denom
    u = 1 - b - d
    return op(b, d, max(u, 0.0), a)


def fuse(w1, w2):
    """R3: cumulative (aleatory) fusion."""
    b1, d1, u1, a1 = w1
    b2, d2, u2, a2 = w2
    k = u1 + u2 - u1 * u2
    if k < 1e-12:  # both dogmatic
        return op((b1 + b2) / 2, (d1 + d2) / 2, 0.0, (a1 + a2) / 2)
    b = (b1 * u2 + b2 * u1) / k
    d = (d1 * u2 + d2 * u1) / k
    u = (u1 * u2) / k
    if abs(u1 - 1) < 1e-12 and abs(u2 - 1) < 1e-12:
        a = (a1 + a2) / 2
    else:
        a = (a1 * u2 + a2 * u1 - (a1 + a2) * u1 * u2) / (u1 + u2 - 2 * u1 * u2) if (u1 + u2 - 2 * u1 * u2) > 1e-12 else a1
    return op(b, d, u, a)


def strip(w):
    """R4: provenance stripping -> vacuous at base rate."""
    return vacuous(w[3])


def conflict(w1, w2):
    """R5: degree of conflict."""
    return abs(P(w1) - P(w2)) * (1 - w1[2]) * (1 - w2[2])


def to_evidence(w):
    """Opinion -> evidence (r, s) under prior weight W."""
    b, d, u, _ = w
    return W * b / u, W * d / u


def from_evidence(r, s, a):
    t = r + s + W
    return op(r / t, s / t, W / t, a)
