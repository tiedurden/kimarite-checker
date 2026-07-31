"""Kimarite taxonomy: 82 official techniques + 7 non-techniques.

Used to (a) normalize/validate labels typed into labels.csv, (b) roll rare
techniques up into their official family for coarse-label training, and
(c) flag pairs that are genuinely indistinguishable at 224x224.

Sanity check: 82 techniques + 7 higi = 89 entries. TECHNIQUE_COUNT asserts it.
"""

# Official category -> {romaji: kanji}. Order matches the sumo association's.
TAXONOMY: dict[str, dict[str, str]] = {
    # 基本技 -- the bread and butter; yorikiri alone is ~32% of all bouts.
    "kihonwaza": {
        "abisetaoshi": "浴びせ倒し", "oshidashi": "押し出し", "oshitaoshi": "押し倒し",
        "tsukidashi": "突き出し", "tsukitaoshi": "突き倒し", "yorikiri": "寄り切り",
        "yoritaoshi": "寄り倒し",
    },
    # 掛け手 -- leg trips.
    "kakete": {
        "ashitori": "足取り", "chongake": "ちょん掛け", "kawazugake": "河津掛け",
        "kekaeshi": "蹴返し", "ketaguri": "蹴手繰り", "kirikaeshi": "切り返し",
        "komatasukui": "小股掬い", "kozumatori": "小褄取り", "mitokorozeme": "三所攻め",
        "nimaigeri": "二枚蹴り", "omata": "大股", "sotogake": "外掛",
        "sotokomata": "外小股", "susoharai": "裾払い", "susotori": "裾取り",
        "tsumatori": "褄取り", "uchigake": "内掛け", "watashikomi": "渡し込み",
    },
    # 投げ手 -- throws.
    "nagete": {
        "ipponzeoi": "一本背負い", "kakenage": "掛け投げ", "koshinage": "腰投げ",
        "kotenage": "小手投げ", "kubinage": "首投げ", "nichonage": "二丁投げ",
        "shitatedashinage": "下手出し投げ", "shitatenage": "下手投げ",
        "sukuinage": "掬い投げ", "tsukaminage": "つかみ投げ",
        "uwatedashinage": "上手出し投げ", "uwatenage": "上手投げ",
        "yaguranage": "櫓投げ",
    },
    # 捻り手 -- twist downs.
    "hinerite": {
        "amiuchi": "網打ち", "gasshohineri": "合掌捻り", "harimanage": "波離間投げ",
        "kainahineri": "腕捻り", "katasukashi": "肩透かし", "kotehineri": "小手捻り",
        "kubihineri": "首捻り", "makiotoshi": "巻き落とし", "osakate": "大逆手",
        "sabaori": "鯖折り", "sakatottari": "逆とったり", "shitatehineri": "下手捻り",
        "sotomuso": "外無双", "tokkurinage": "徳利投げ", "tottari": "とったり",
        "tsukiotoshi": "突き落とし", "uchimuso": "内無双", "uwatehineri": "上手捻り",
        "zubuneri": "ずぶねり",
    },
    # 反り手 -- backward body drops. Spectacular and vanishingly rare; tasukizori
    # went 65 years between occurrences. Expect zero examples of most of these.
    "sorite": {
        "izori": "居反り", "kakezori": "掛け反り", "shumokuzori": "撞木反り",
        "sototasukizori": "外たすき反り", "tasukizori": "たすき反り",
        "tsutaezori": "伝え反り",
    },
    # 特殊技 -- everything else, including the very common hatakikomi.
    "tokushuwaza": {
        "hatakikomi": "叩き込み", "hikiotoshi": "引き落とし", "hikkake": "引っ掛け",
        "kimedashi": "極め出し", "kimetaoshi": "極め倒し", "okuridashi": "送り出し",
        "okurigake": "送り掛け", "okurihikiotoshi": "送り引き落とし",
        "okurinage": "送り投げ", "okuritaoshi": "送り倒し",
        "okuritsuridashi": "送り吊り出し", "okuritsuriotoshi": "送り吊り落とし",
        "sokubiotoshi": "素首落とし", "tsuridashi": "吊り出し",
        "tsuriotoshi": "吊り落とし", "ushiromotare": "後ろもたれ",
        "utchari": "うっちゃり", "waridashi": "割り出し", "yobimodoshi": "呼び戻し",
    },
}

# 非技 -- NOT techniques. The loser lost unaided; there is no causal motion to
# learn. `fusen` is worst: the opponent never appeared, so no bout was fought and
# any clip cut for it shows unrelated footage. Excluded from training by default.
NON_TECHNIQUES: dict[str, str] = {
    "fumidashi": "踏み出し", "isamiashi": "勇み足", "koshikudake": "腰砕け",
    "tsukihiza": "つきひざ", "tsukite": "つき手", "fusen": "不戦", "hansoku": "反則",
}

TECHNIQUE_COUNT = 82

# Pairs that differ ONLY in how the bout ENDED -- same technique, opponent
# standing (-dashi/-kiri) vs fallen (-taoshi/-otoshi). The distinguishing evidence
# is the last ~0.5s. If segment_bouts.py trims the fall, these are unlearnable
# and the model can only ever guess the more common member.
OUTCOME_PAIRS = [
    ("oshidashi", "oshitaoshi"), ("tsukidashi", "tsukitaoshi"),
    ("yorikiri", "yoritaoshi"), ("kimedashi", "kimetaoshi"),
    ("tsuridashi", "tsuriotoshi"), ("okuridashi", "okuritaoshi"),
    ("okuritsuridashi", "okuritsuriotoshi"),
]

# Pairs differing only by grip/limb SIDE: uwate = over/outside the opponent's
# arm, shitate = under/inside; soto = outside, uchi = inside. A few centimetres
# of arm position, often occluded by two bodies. Do not expect to separate these
# at 224x224 -- persistent confusion here is a resolution limit, not a bug.
GRIP_PAIRS = [
    ("uwatenage", "shitatenage"), ("uwatehineri", "shitatehineri"),
    ("uwatedashinage", "shitatedashinage"), ("sotogake", "uchigake"),
    ("sotomuso", "uchimuso"), ("sotokomata", "komatasukui"),
]

CATEGORY_OF: dict[str, str] = {
    name: cat for cat, members in TAXONOMY.items() for name in members
}
ALL_TECHNIQUES = sorted(CATEGORY_OF)
KANJI_OF: dict[str, str] = {
    n: k for members in TAXONOMY.values() for n, k in members.items()
} | dict(NON_TECHNIQUES)

# Long vowels, apostrophes and hyphens vary by romanization (omata/ōmata,
# gasshohineri/gasshōhineri). Fold them so typed labels match.
_FOLD = str.maketrans({"ō": "o", "ū": "u", "ā": "a", "ē": "e", "ī": "i",
                       "'": "", "-": "", " ": "", "_": ""})


def normalize(raw: str) -> str | None:
    """Map a typed or kanji label to a canonical romaji key, or None if unknown."""
    s = (raw or "").strip()
    if not s:
        return None
    if s in KANJI_OF.values():  # someone pasted the kanji
        return next(k for k, v in KANJI_OF.items() if v == s)
    key = s.lower().translate(_FOLD)
    if key in CATEGORY_OF or key in NON_TECHNIQUES:
        return key
    # Tolerate a trailing/leading typo of one char before giving up.
    for cand in list(CATEGORY_OF) + list(NON_TECHNIQUES):
        if key.startswith(cand) or cand.startswith(key):
            if abs(len(key) - len(cand)) <= 2:
                return cand
    return None


def is_technique(name: str) -> bool:
    """False for higi (non-techniques) -- exclude these from training."""
    return name in CATEGORY_OF


def coarse(name: str) -> str | None:
    """Official family, for coarse-label training. None for non-techniques."""
    return CATEGORY_OF.get(name)


def partner(name: str) -> str | None:
    """The confusable counterpart, if this technique has one."""
    for a, b in OUTCOME_PAIRS + GRIP_PAIRS:
        if name == a:
            return b
        if name == b:
            return a
    return None


if __name__ == "__main__":
    n = len(CATEGORY_OF)
    assert n == TECHNIQUE_COUNT, f"expected {TECHNIQUE_COUNT} techniques, got {n}"
    assert not (set(CATEGORY_OF) & set(NON_TECHNIQUES)), "overlap in vocabularies"
    print(f"{n} techniques + {len(NON_TECHNIQUES)} non-techniques "
          f"= {n + len(NON_TECHNIQUES)} labels")
    for cat, members in TAXONOMY.items():
        print(f"  {cat:<14} {len(members):>3}")
    print(f"\n{len(OUTCOME_PAIRS)} outcome pairs (need the bout's final moment)")
    print(f"{len(GRIP_PAIRS)} grip pairs (likely unresolvable at 224x224)")
