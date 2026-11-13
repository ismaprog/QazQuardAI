#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ICT1103 · QAIRU · 2026–2027
Екінші аралық бақылау: топтық жоба прототипі 
Тақырыбы: QazGuard AI — корпоративтік жүйелер мен сыртқы LLM арасындағы шлюз:
құпия деректерді (ЖСН, БСН, IBAN, карта, телефон, email) токендеу және Jailbreak жіктеу.

Негізгі тақырыптармен байланыс:
- 5-апта: киберқауіпсіздік (қатер моделі, DLP/AI-шлюз)
- 6-апта: оқытушымен оқыту (жіктеу, train/test, метрикалар, қателер матрицасы)
- 7-апта: NLP (TF-IDF, мәтінді жіктеу)

Тек стандартты Python, NumPy және Pandas қолданылады.
"""

import os
import re
import math
import random
import secrets
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
DATASET_PATH = os.path.join(OUTPUT_DIR, "synthetic_prompts_kz.csv")
LABELS = ["SAFE", "SENSITIVE_PII", "JAILBREAK"]


# ==============================================================================
# 0. БАҚЫЛАУ СОМАЛАРЫ (ЖСН/БСН, IBAN, Luhn)
# ==============================================================================
def _weight2(i: int) -> int:
    """ЖСН/БСН екінші өту салмақтары (0..10 позициялар үшін): 3,4,...,11,1,2."""
    w = i + 3
    return w if w <= 11 else w - 11


def _id_check_digit(first11: str):
    d = [int(c) for c in first11]
    s = sum(d[i] * (i + 1) for i in range(11)) % 11
    if s == 10:
        s = sum(d[i] * _weight2(i) for i in range(11)) % 11
    return None if s == 10 else s


def id_checksum_ok(num: str) -> bool:
    if not (len(num) == 12 and num.isdigit()):
        return False
    c = _id_check_digit(num[:11])
    return c is not None and c == int(num[11])


def is_valid_iin(num: str) -> bool:
    """ЖСН: бақылау соммасы + ай (01-12) + күн (01-31) + ғасыр/жыныс цифры (1-6)."""
    if not id_checksum_ok(num):
        return False
    mm, dd = int(num[2:4]), int(num[4:6])
    return 1 <= mm <= 12 and 1 <= dd <= 31 and num[6] in "123456"


def is_valid_bin(num: str) -> bool:
    """БСН: бақылау соммасы + 5-цифр (заңды тұлға түрі) 4, 5 немесе 6."""
    return id_checksum_ok(num) and num[4] in "456"


def iban_ok(iban: str) -> bool:
    iban = iban.replace(" ", "").upper()
    if not re.fullmatch(r"KZ\d{2}[0-9A-Z]{16}", iban):
        return False
    rearr = iban[4:] + iban[:4]
    return int("".join(str(int(ch, 36)) for ch in rearr)) % 97 == 1


def luhn_ok(card: str) -> bool:
    digits = [int(c) for c in re.sub(r"\D", "", card)]
    if len(digits) != 16:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0
