"""
allergens.py - the safety layer.

No AI in this file, on purpose. A language model proposes meals; this module
decides whether they are allowed. It is a word-list matcher, so it is
predictable, testable and fast. It is deliberately biased toward over-blocking:
a wrongly rejected meal costs one retry, a missed allergen costs far more.

It is NOT a medical device. It cannot see inside packaged food or catch
cross-contact. See the README.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


# --------------------------------------------------------------------------
# Text normalisation: lowercase, strip accents and apostrophes, punctuation -> space
# --------------------------------------------------------------------------
def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    text = re.sub(r"['\u2019`]", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _rx(terms) -> re.Pattern:
    """Whole-word match, tolerating a plural -s/-es. 'eggplant' must not match 'egg'."""
    alts = sorted({re.escape(normalise(t)) for t in terms}, key=len, reverse=True)
    return re.compile(r"(?<![a-z0-9])(?:" + "|".join(alts) + r")(?:s|es)?(?![a-z0-9])")


# --------------------------------------------------------------------------
# Allergen knowledge base. Includes hidden sources and a few regional terms.
# --------------------------------------------------------------------------
_WHEAT = [
    "wheat", "wholewheat", "wheatgerm", "flour", "bread", "breadcrumb", "panko", "toast",
    "sourdough", "flatbread", "cornbread", "shortbread", "gingerbread", "bagel", "bun",
    "croissant", "brioche", "pita", "naan", "roti", "chapati", "paratha", "puri", "kulcha",
    "tortilla", "wrap", "cracker", "biscuit", "cookie", "cake", "muffin", "pancake", "waffle",
    "pastry", "dough", "pizza", "sandwich", "pasta", "spaghetti", "linguine", "fettuccine",
    "penne", "macaroni", "lasagna", "lasagne", "ravioli", "tortellini", "gnocchi", "orzo",
    "vermicelli", "noodle", "ramen", "udon", "soba", "couscous", "bulgur", "semolina",
    "durum", "spelt", "farro", "kamut", "freekeh", "seitan", "dumpling", "wonton", "gyoza",
    "samosa", "tempura", "batter", "atta", "maida", "suji", "sooji", "rava", "dalia", "bran",
    "graham", "soy sauce", "teriyaki", "hoisin",
]

ALLERGENS: dict[str, list[str]] = {
    "milk": [
        "milk", "cream", "butter", "ghee", "cheese", "cheddar", "mozzarella", "parmesan",
        "parmigiano", "pecorino", "feta", "ricotta", "mascarpone", "halloumi", "gruyere",
        "gouda", "brie", "camembert", "provolone", "paneer", "yogurt", "yoghurt", "curd",
        "dahi", "kefir", "skyr", "quark", "buttermilk", "whey", "casein", "caseinate",
        "lactose", "lactalbumin", "custard", "ice cream", "gelato", "sour cream",
        "creme fraiche", "half and half", "condensed milk", "evaporated milk", "milk powder",
        "milkshake", "dairy", "pesto", "khoya", "mawa", "malai", "lassi", "raita", "kheer",
        "rabdi", "nougat",
    ],
    "egg": [
        "egg", "yolk", "albumin", "albumen", "ovalbumin", "lysozyme", "mayonnaise", "mayo",
        "aioli", "meringue", "eggnog", "hollandaise", "egg wash",
    ],
    "peanut": [
        "peanut", "groundnut", "arachis", "goober", "satay", "nut", "moongphali", "mungfali",
    ],
    "tree_nut": [
        "almond", "walnut", "cashew", "pecan", "pistachio", "hazelnut", "filbert", "macadamia",
        "brazil nut", "chestnut", "pine nut", "pignoli", "marzipan", "praline", "nougat",
        "gianduja", "nutella", "frangipane", "amaretto", "pesto", "nut butter", "nut milk",
        "nut flour", "mixed nut", "tree nut", "nut", "kaju", "badam", "pista", "akhrot",
    ],
    "crustacean": [
        "shrimp", "prawn", "crab", "lobster", "crayfish", "crawfish", "langoustine", "krill",
        "scampi", "shrimp paste", "belacan", "crustacean", "shellfish",
    ],
    "mollusc": [
        "clam", "mussel", "oyster", "scallop", "squid", "calamari", "octopus", "cuttlefish",
        "abalone", "escargot", "snail", "whelk", "cockle", "mollusc", "mollusk",
        "oyster sauce", "shellfish",
    ],
    "fish": [
        "fish", "salmon", "tuna", "cod", "haddock", "trout", "anchovy", "anchovies",
        "sardine", "mackerel", "tilapia", "halibut", "snapper", "herring", "pollock",
        "swordfish", "catfish", "monkfish", "mahi mahi", "surimi", "caviar", "roe", "bonito",
        "katsuobushi", "dashi", "worcestershire", "fish sauce", "caesar dressing", "rohu",
        "hilsa", "ilish", "pomfret", "kingfish", "surmai", "katla", "bhetki",
    ],
    "wheat": _WHEAT,
    "gluten": _WHEAT + ["barley", "rye", "malt", "malted", "malt vinegar", "beer", "ale",
                        "lager", "triticale", "brewers yeast"],
    "soy": [
        "soy", "soya", "soybean", "edamame", "tofu", "tempeh", "miso", "soy sauce", "tamari",
        "shoyu", "natto", "tvp", "textured vegetable protein", "soy lecithin", "bean curd",
        "yuba", "hoisin", "teriyaki", "nutrela",
    ],
    "sesame": [
        "sesame", "tahini", "hummus", "halva", "halvah", "gomasio", "benne", "til",
        "gingelly", "zaatar", "zatar", "dukkah",
    ],
    "mustard": ["mustard", "sarson"],
    "celery": ["celery", "celeriac"],
}

# Ingredients that are *probably* fine for a category but that some people with the
# allergy avoid. Reported as "caution" (ask the person); blocking if strict=True.
CAUTION: dict[str, list[str]] = {
    "tree_nut": ["coconut", "shea"],
    "peanut": ["lupin", "lupine"],
    "gluten": ["oat"],
    "soy": ["lecithin"],
}

# Phrases scrubbed before matching a category, to avoid obvious false positives.
_GLUTEN_FREE = [
    "gluten free pasta", "gluten free bread", "gluten free flour", "gluten free noodle",
    "gluten free tortilla", "gluten free oat", "gluten free soy sauce", "gluten free cracker",
    "rice noodle", "glass noodle", "konjac noodle", "shirataki noodle", "sweet potato noodle",
    "rice cake", "rice cracker", "corn tortilla", "rice flour", "almond flour", "coconut flour",
    "chickpea flour", "gram flour", "besan", "corn flour", "cornflour", "tapioca flour",
    "potato flour", "cassava flour", "buckwheat flour", "oat flour", "masa harina",
    "pasta sauce", "tomato pasta sauce",
]
EXCEPTIONS: dict[str, list[str]] = {
    "milk": [
        "peanut butter", "almond butter", "cashew butter", "nut butter", "seed butter",
        "sunflower butter", "cocoa butter", "shea butter", "apple butter", "coconut butter",
        "butter bean", "butter lettuce", "coconut milk", "coconut cream", "cream of coconut",
        "cream of tartar", "almond milk", "oat milk", "soy milk", "rice milk", "cashew milk",
        "hemp milk", "pea milk", "plant milk", "nut milk", "coconut yogurt",
    ],
    "egg": ["flax egg", "chia egg", "egg replacer"],
    "tree_nut": ["water chestnut", "nut free", "peanut"],
    "peanut": ["nut free", "tree nut"],
    "wheat": _GLUTEN_FREE,
    "gluten": _GLUTEN_FREE,
}
# "dairy free", "nut free", ... : scrubbed for every category.
_FREE_OF = re.compile(
    r"(?<![a-z0-9])(?:dairy|milk|lactose|egg|nut|peanut|tree nut|soy|gluten|wheat|fish|"
    r"shellfish|sesame) free(?![a-z0-9])"
)

# Words that mean "processed product: allergens depend on the brand". Reminders only.
LABEL_CHECK = [
    "stock", "broth", "bouillon", "seasoning", "spice mix", "spice blend", "curry paste",
    "curry powder", "garam masala", "masala", "sauce", "dressing", "gravy", "marinade",
    "chocolate", "sausage", "chips", "crisps", "cereal", "granola", "protein bar",
    "vegan cheese", "vegan butter", "plant based", "non dairy", "baking mix", "ketchup",
    "vinegar", "tinned", "canned", "jarred", "packaged", "ready made", "store bought",
]

# What people write in a profile -> internal categories.
ALIASES: dict[str, list[str]] = {
    "dairy": ["milk"], "milk": ["milk"], "lactose": ["milk"], "cow milk": ["milk"],
    "egg": ["egg"], "eggs": ["egg"],
    "peanut": ["peanut"], "peanuts": ["peanut"], "groundnut": ["peanut"], "groundnuts": ["peanut"],
    "tree nut": ["tree_nut"], "tree nuts": ["tree_nut"], "treenut": ["tree_nut"],
    "nuts": ["peanut", "tree_nut"], "nut": ["peanut", "tree_nut"],
    "shellfish": ["crustacean", "mollusc"], "crustacean": ["crustacean"],
    "crustaceans": ["crustacean"], "mollusc": ["mollusc"], "molluscs": ["mollusc"],
    "mollusk": ["mollusc"], "mollusks": ["mollusc"],
    "fish": ["fish"], "seafood": ["fish", "crustacean", "mollusc"],
    "wheat": ["wheat"], "gluten": ["gluten"], "celiac": ["gluten"], "coeliac": ["gluten"],
    "soy": ["soy"], "soya": ["soy"], "soybean": ["soy"], "soybeans": ["soy"],
    "sesame": ["sesame"], "mustard": ["mustard"], "celery": ["celery"],
}

_LABELS = {
    "tree_nut": "tree nut", "crustacean": "crustacean shellfish", "mollusc": "mollusc",
}

_CATEGORY_RX = {k: _rx(v) for k, v in ALLERGENS.items()}
_CAUTION_RX = {k: _rx(v) for k, v in CAUTION.items()}
_EXC_RX = {k: [_rx([p]) for p in v] for k, v in EXCEPTIONS.items()}
_LABEL_RX = _rx(LABEL_CHECK)


@dataclass(frozen=True)
class Hit:
    allergen: str  # category name, or the literal word for a custom avoid
    term: str      # the word that matched
    text: str      # the line it was found in
    level: str     # "block" or "caution"

    def __str__(self) -> str:
        what = "may relate to" if self.level == "caution" else "is"
        return f"'{self.term}' {what} {self.allergen.replace('_', ' ')} (in: \"{self.text}\")"


class AllergenGuard:
    def __init__(self, allergies, also_avoid=(), strict: bool = False):
        self.categories: list[str] = []
        self.custom: list[str] = []
        self.unrecognised: list[str] = []
        self.strict = strict

        for raw in allergies:
            key = normalise(raw)
            cats = ALIASES.get(key) or ALIASES.get(key.rstrip("s"))
            if cats:
                for c in cats:
                    if c not in self.categories:
                        self.categories.append(c)
            elif key:
                # Never silently drop an allergy: treat it as a literal word to avoid.
                self.custom.append(key)
                self.unrecognised.append(str(raw))
        for raw in also_avoid:
            key = normalise(raw)
            if key and key not in self.custom:
                self.custom.append(key)
        self._custom_rx = {c: _rx([c]) for c in self.custom}

    # -- scanning ----------------------------------------------------------
    def scan(self, text: str) -> list[Hit]:
        norm = normalise(text)
        hits: dict[tuple[str, str], Hit] = {}

        for cat in self.categories:
            scrubbed = norm
            for rx in _EXC_RX.get(cat, []):  # e.g. "gluten free pasta", "coconut milk"
                scrubbed = rx.sub(" ", scrubbed)
            scrubbed = _FREE_OF.sub(" ", scrubbed)  # then "dairy free", "nut free"
            for m in _CATEGORY_RX[cat].finditer(scrubbed):
                hits.setdefault((cat, m.group(0)), Hit(cat, m.group(0), str(text), "block"))
            if cat in _CAUTION_RX:
                level = "block" if self.strict else "caution"
                for m in _CAUTION_RX[cat].finditer(scrubbed):
                    hits.setdefault((cat, m.group(0)), Hit(cat, m.group(0), str(text), level))

        for word, rx in self._custom_rx.items():
            for m in rx.finditer(norm):
                hits.setdefault((word, m.group(0)), Hit(word, m.group(0), str(text), "block"))

        return list(hits.values())

    def label_flags(self, text: str) -> list[str]:
        return sorted({m.group(0) for m in _LABEL_RX.finditer(normalise(text))})

    def describe(self) -> str:
        parts = [_LABELS.get(c, c.replace("_", " ")) for c in self.categories]
        parts += self.custom
        return ", ".join(parts) if parts else "nothing"
