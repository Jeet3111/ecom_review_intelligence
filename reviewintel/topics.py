"""
topics.py - lightweight, explainable topic (aspect) detection by keyword lexicon.
Used for the 'Top positive / negative topics' dashboard and for the web app.
(This is a rule-based component, not part of the neural network.)
"""
import re

TOPICS = {
    "Delivery & logistics": [r"deliver", r"courier", r"shipping", r"arrived", r"dispatch", r"late\b", r"delay"],
    "Packaging": [r"packag", r"packing", r"packed", r"\bbox\b", r"sealed", r"bubble", r"torn"],
    "Product quality": [r"quality", r"sturdy", r"premium", r"cheap", r"material", r"broke", r"broken", r"build"],
    "Value for money": [r"price", r"paisa", r"money", r"worth", r"overpriced", r"discount", r"rupee"],
    "Customer service": [r"customer care", r"customer service", r"refund", r"replacement", r"support", r"complaint"],
    "Battery & charging": [r"battery", r"charging", r"\bcharge", r"backup"],
    "Camera & display": [r"camera", r"photo", r"display", r"screen", r"pixel", r"brightness"],
    "Performance & sound": [r"\blag\b", r"hang", r"crash", r"performance", r"sound", r"bass", r"bluetooth", r"connect"],
    "Fit, fabric & colour": [r"\bsize\b", r"\bfit\b", r"fabric", r"stitch", r"colou?r", r"tight", r"loose", r"thread"],
    "Skin, shade & texture": [r"\bskin\b", r"shade", r"texture", r"fragrance", r"smell", r"greasy", r"rash", r"pimple"],
    "Taste & freshness": [r"taste", r"fresh", r"expiry", r"stale", r"quantity", r"rotten", r"tasty"],
    "Ease of use & durability": [r"easy to", r"clean", r"instruction", r"assembly", r"handle", r"durab", r"setup"],
}
_COMPILED = {k: re.compile("|".join(v), re.I) for k, v in TOPICS.items()}


def detect_topics(text):
    return [k for k, rx in _COMPILED.items() if rx.search(str(text))]
