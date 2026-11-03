"""
synth.py - Synthetic "Indian e-commerce review" dataset generator.

WHY SYNTHETIC?  The project must run offline and the real Flipkart / Amazon.in review
dumps cannot be redistributed.  The generator produces realistic English + Hinglish
reviews for six product categories, with ratings, prices, dates and a spam / fake label.
Plug in a real CSV instead with:  python run_pipeline.py --data my_reviews.csv
(see README -> "Using a real dataset").

Spam / fake-review types simulated (about 20% of rows):
    promo_url, paid_review, gibberish, offtopic, generic_fake_praise, hybrid
Several genuine "hard negatives" (genuine reviews that mention phone numbers, WhatsApp,
'5 star', 'like my review' ...) and subtle spam (no link at all) are included on purpose so
the task is not trivially separable.
"""
import random
import string
import numpy as np
import pandas as pd

# ------------------------------------------------------------------ catalogue
CATEGORIES = {
    "Mobiles & Accessories": dict(price=(6500, 48000), platform=["Flipkart", "Amazon.in"],
                                  brands=["Nova", "Zenith", "Pixa", "Orbit", "Vega", "Lumo"],
                                  items=["X5 5G", "Lite 4G", "Pro Max", "S2 Smartphone", "Note 12", "Neo 5G"],
                                  aspects=["battery", "camera", "display", "performance"]),
    "Electronics": dict(price=(1200, 55000), platform=["Amazon.in", "Flipkart"],
                        brands=["Sonic", "Bolt", "Aura", "Techno", "Pulse", "Kinetic"],
                        items=["Wireless Earbuds", "Bluetooth Speaker", "Smart TV 43\"", "Laptop 15", "Smart Watch", "Soundbar"],
                        aspects=["sound", "connectivity", "battery", "performance"]),
    "Fashion": dict(price=(399, 4500), platform=["Myntra", "Flipkart", "Meesho"],
                    brands=["UrbanFit", "Desi Threads", "Kurtiwala", "StreetKart", "Rangrez", "Mehfil"],
                    items=["Cotton Kurta", "Slim Jeans", "Running Shoes", "Saree", "Hoodie", "Formal Shirt"],
                    aspects=["fit", "fabric", "colour", "stitching"]),
    "Home & Kitchen": dict(price=(250, 12000), platform=["Amazon.in", "Flipkart", "Meesho"],
                           brands=["HomeEase", "Chulha", "Aqua", "Kitchenova", "Gharwali", "PrimeHome"],
                           items=["Mixer Grinder", "Pressure Cooker", "Non-stick Tawa", "Water Bottle", "Air Fryer", "Dinner Set"],
                           aspects=["durability", "ease_of_use", "cleaning", "performance"]),
    "Beauty & Personal Care": dict(price=(149, 2500), platform=["Nykaa", "Amazon.in", "Flipkart"],
                                   brands=["GlowUp", "Ayurveda Plus", "Veda", "Petal", "Naturelle", "Kesar"],
                                   items=["Face Serum", "Matte Lipstick", "Sunscreen SPF 50", "Hair Oil", "Foundation", "Moisturiser"],
                                   aspects=["skin", "fragrance", "texture", "shade"]),
    "Grocery & Gourmet": dict(price=(60, 1800), platform=["JioMart", "Amazon.in", "Flipkart"],
                              brands=["Annapurna", "Swad", "FreshFarm", "Desi Dhaba", "Masala House", "Golden Grain"],
                              items=["Basmati Rice 5kg", "Masala Chai", "Cold Pressed Oil", "Dry Fruits Mix", "Millet Cookies", "Ghee 1L"],
                              aspects=["taste", "freshness", "quantity", "expiry"]),
}

# ------------------------------------------------------------------ aspects
# noun, 2 positive phrases, 2 negative phrases
ASPECTS = {
    # universal
    "delivery": ("delivery", ["Got it delivered two days before the promised date.", "The delivery executive was polite and the parcel arrived on time.", "Delivery ekdum time pe ho gayi."],
                 ["Delivery took more than ten days and nobody picked my calls.", "The courier kept delaying and finally marked it undelivered.", "Delivery late ho gayi, very bad experience."]),
    "packaging": ("packaging", ["Packed very securely with bubble wrap.", "Box was sealed properly, no damage at all."],
                  ["The box was torn and the product looked used.", "Packaging was so poor that the item arrived damaged.", "Packing bahut kharab thi."]),
    "quality": ("product quality", ["Quality is far better than what I expected at this price.", "Feels premium and sturdy in hand."],
                ["Quality is very cheap, started breaking within a week.", "Looks nothing like the pictures, poor material."]),
    "price": ("price", ["Totally paisa vasool at this price.", "Got it in the Big Billion sale at a great discount."],
              ["Overpriced for what you actually get.", "Not worth the money, paisa barbaad."]),
    "service": ("customer service", ["Customer care resolved my issue in a single call.", "Replacement was arranged quickly without any fuss."],
                ["Customer care is useless, no response to my complaint.", "Refund still not received after three weeks."]),
    # mobiles / electronics
    "battery": ("battery backup", ["Battery easily lasts a full day with heavy usage.", "Charging is really fast, full in about an hour."],
                ["Battery drains by evening even with light use.", "Heats up while charging and battery dies very fast."]),
    "camera": ("camera", ["Camera captures sharp photos even in low light.", "Photos and videos come out crisp and detailed."],
               ["Camera quality is grainy and the focus is slow.", "Camera photos look washed out, very disappointing."]),
    "display": ("display", ["Display is bright and the colours look vivid.", "Screen is smooth and sharp, great for videos."],
                ["Screen has dead pixels and looks dull outdoors.", "Display brightness is very low."]),
    "performance": ("performance", ["No lag at all, handles games smoothly.", "Performance is snappy and apps open instantly."],
                    ["Hangs a lot and apps keep crashing.", "Very slow performance, speed is poor."]),
    "sound": ("sound quality", ["Bass is deep and the sound is crystal clear.", "Sound quality is rich and loud."],
              ["Sound is tinny and there is a constant crackling noise.", "Sound quality is flat, no bass at all."]),
    "connectivity": ("bluetooth connectivity", ["Bluetooth pairs instantly and never disconnects.", "Connection is stable even from another room."],
                     ["Keeps disconnecting every few minutes.", "Bluetooth connection drops constantly."]),
    # fashion
    "fit": ("size and fit", ["Size is perfect, fits true to the chart.", "The fit is comfortable and looks great."],
            ["Too tight, order one size up. Size chart is wrong.", "Fit is terrible, very loose at the waist."]),
    "fabric": ("fabric", ["Fabric is soft and breathable, great for summers.", "Cotton is thick and feels premium."],
               ["Fabric is rough and feels very cheap.", "Fabric is thin and see-through."]),
    "colour": ("colour", ["Colour is exactly as shown in the pictures.", "The colour looks rich and does not fade."],
               ["Colour faded after the very first wash.", "Colour is completely different from the photo."]),
    "stitching": ("stitching", ["Stitching is neat and strong.", "Finishing and stitching are top class."],
                  ["Threads were already coming out, stitching is poor.", "Stitching came apart in two washes."]),
    # home & kitchen
    "durability": ("durability", ["Been using it daily for three months, still like new.", "Very durable and solidly built."],
                   ["The handle broke within two weeks.", "Not durable at all, stopped working after a month."]),
    "ease_of_use": ("ease of use", ["Very easy to use, even my mother figured it out.", "Setup was simple and the instructions are clear."],
                    ["Instructions are confusing and assembly was a nightmare.", "Difficult to use, the buttons are stiff."]),
    "cleaning": ("cleaning", ["Easy to clean and maintain.", "Wipes clean in seconds."],
                 ["Hard to clean, food sticks everywhere.", "Cleaning is a pain, the parts do not come apart."]),
    # beauty
    "skin": ("skin feel", ["Suits my sensitive skin, no breakouts at all.", "My skin feels soft and glowing after a week."],
             ["Caused rashes and pimples on my skin.", "Irritated my skin badly, had to stop using it."]),
    "fragrance": ("fragrance", ["Smells lovely and lasts for hours.", "The fragrance is mild and refreshing."],
                  ["Fragrance is far too strong and gave me a headache.", "Smells chemical and unpleasant."]),
    "texture": ("texture", ["Texture is light and non-greasy.", "Texture is silky and absorbs quickly."],
                ["Texture is sticky and very greasy.", "The texture is runny and messy."]),
    "shade": ("shade", ["Shade matches my skin tone perfectly.", "Shade is exactly as shown on the website."],
              ["Shade is completely different from what is shown.", "Wrong shade, looks patchy on my skin."]),
    # grocery
    "taste": ("taste", ["Tastes fresh and authentic, just like home.", "The taste is wonderful, whole family loved it."],
              ["Taste is bland and stale.", "Tastes bitter, nothing like the brand I know."]),
    "freshness": ("freshness", ["Everything was fresh and well packed.", "Very fresh stock, no complaints."],
                  ["Items were rotten and wilted.", "Not fresh at all, smelled bad on opening."]),
    "quantity": ("quantity", ["Quantity is generous for the price.", "Got more quantity than I expected."],
                 ["Quantity is much less than what is mentioned.", "The pack was only half full."]),
    "expiry": ("expiry date", ["Long expiry date, got a fresh batch.", "Expiry is more than a year away, good."],
               ["Product was close to its expiry date.", "Received an expired item, very dangerous."]),
}
UNIVERSAL = ["delivery", "packaging", "quality", "price", "service"]

POS_ADJ = ["excellent", "superb", "really good", "amazing", "very impressive", "top notch", "awesome", "great"]
NEG_ADJ = ["terrible", "very poor", "disappointing", "the worst", "pathetic", "bad", "not good at all", "awful"]
NEU_ADJ = ["okay", "average", "decent", "acceptable", "fine but nothing special", "just alright"]
POS_HI = ["mast", "badhiya", "zabardast", "ek number"]
NEG_HI = ["bakwas", "kharab", "ghatiya", "bekaar"]
NEU_HI = ["theek-thaak", "chalega", "average"]

OPEN_POS = ["Very happy with this purchase.", "Totally satisfied!", "Loved it.", "Worth every rupee.", "", "", ""]
OPEN_NEG = ["Very disappointed.", "Do not buy this.", "Waste of money.", "Pathetic experience.", "Paisa barbaad.", "", "", ""]
OPEN_NEU = ["Overall an average buy.", "Okay for the price.", "Could be better.", "", "", ""]
CLOSE_POS = ["Highly recommended.", "Go for it.", "Will buy again.", "Thank you seller.", "Must buy!", "", "", ""]
CLOSE_NEG = ["Never buying from this seller again.", "Please improve.", "Want my money back.", "Avoid.", "", "", ""]
CLOSE_NEU = ["Let us see how it lasts.", "Decent for now.", "", "", ""]
SHORT = {"pos": ["Good product", "Nice", "Value for money", "Awesome product, must buy", "Very good", "Superb quality", "Excellent, loved it", "Best in this price range"],
         "neg": ["Worst product ever", "Not worth it", "Very bad quality", "Totally waste", "Poor product, don't buy", "Bakwas product", "Disappointed"],
         "neu": ["Okay product", "Average", "Not bad not great", "Just ok", "Decent for the price"]}
# genuine messages that *look* spammy (hard negatives)
HARD_GENUINE = ["I called customer care on 1800-{d3}-{d4} but nobody helped.", "Order id {oid}, please check the delivery status.",
                "Hope this review helps, please like if useful.", "Giving 5 stars because it works fine.",
                "Contacted the seller on WhatsApp and he replied quickly.", "Visit the service centre if you face any issue.",
                "My brother bought the same one last year and it is still working.", "Will update my review after one month of use!!",
                "Do not buy!!! Total fraud!!!", "BEST PRODUCT, VERY HAPPY"]
EMOJIS = [" \U0001F44D", " \U0001F60A", " \U0001F621", " \u2B50\u2B50\u2B50\u2B50\u2B50", " \U0001F44C", " \U0001F4AF"]

# ------------------------------------------------------------------ spam banks
SP_HOOK = ["Get 90% OFF on all mobiles today only!", "Free recharge of Rs 500, click now.", "Earn Rs 5000 daily from home, no investment needed!",
           "Win a free iPhone in our lucky draw.", "Cheap original products, limited stock.", "Loan approved in 5 minutes without documents.",
           "Work from home and earn money, apply now.", "Buy followers and likes at lowest rates."]
SP_PAID = ["Seller is giving Rs 100 refund for a 5 star review. DM me for details.", "Paid review service, earn money by writing reviews, join our telegram.",
           "I got this product free for giving 5 star rating, you can also get it, message me.", "Contact me for guaranteed positive reviews at a low cost.",
           "Need 5 star ratings for your product? we provide genuine looking reviews.", "Refund after review. Send screenshot of your 5 star review to get cashback."]
SP_OFF = ["India will win the match today, bet on our site and double your money.", "Please subscribe to my YouTube channel, I post daily.",
          "Follow me on Instagram for fashion tips and giveaways.", "Check out my new song on YouTube and share it.",
          "Vote for my candidate in the college election.", "Looking for a roommate in Pune, contact me."]
SP_FAKE = ["Best product ever!!! Must buy!!! 5 star 5 star 5 star", "Awesome awesome product, super quality, everyone should buy, highly recommended, 100% genuine product",
           "Nice product nice nice nice, thank you seller thank you, best seller best product best price", "Wow excellent product, amazing, superb, outstanding. Buy now before stock ends!!!",
           "Very good product very good seller very good delivery very good quality very good", "Perfect perfect perfect product. Five star. Must buy. Don't think just buy."]
SP_TAIL = ["Also join our group for free recharge: {url}", "PS: follow me for more deals {handle}", "Like my review and I will send you a free gift.",
           "Check my profile for discount coupons.", "Contact {phone} for bulk orders at half price.", "Visit {url} for cheaper prices."]
SP_SUBTLE = ["Best product in the world, everyone must buy.", "100% genuine product, totally satisfied, 5 stars.", "Superb, simply the best. Do not think twice.",
             "Highly recommended to all my friends and family.", "Best seller, best product, best price.", "", "", ""]
KEYB = ["asdfghjkl", "qwertyuiop", "zxcvbnm", "lkjhgfdsa", "poiuytrewq"]


def _phone(r):
    return str(r.choice([6, 7, 8, 9])) + "".join(str(r.randint(0, 9)) for _ in range(9))


def _url(r):
    w = "".join(r.choice(string.ascii_lowercase) for _ in range(r.randint(5, 8)))
    return r.choice([f"www.{w}-deals.in", f"bit.ly/{w[:4]}{r.randint(10, 99)}", f"http://{w}.co.in/offer", f"wa.me/91{_phone(r)}"])


def _handle(r):
    return "@" + "".join(r.choice(string.ascii_lowercase) for _ in range(r.randint(5, 9))) + str(r.randint(1, 99))


# ------------------------------------------------------------------ text builders
def _aspect_sentence(r, key, pol, hinglish):
    noun, pos, neg = ASPECTS[key]
    if pol == "pos" and r.random() < 0.45:
        return r.choice(pos)
    if pol == "neg" and r.random() < 0.45:
        return r.choice(neg)
    if hinglish:
        adj = r.choice({"pos": POS_HI, "neg": NEG_HI, "neu": NEU_HI}[pol])
        return f"{noun.capitalize()} ekdum {adj} hai."
    adj = r.choice({"pos": POS_ADJ, "neg": NEG_ADJ, "neu": NEU_ADJ}[pol])
    return r.choice([f"The {noun} is {adj}.", f"{noun.capitalize()} is {adj}!", f"Honestly, the {noun} is {adj}.", f"{noun.capitalize()}: {adj}."])


def _genuine_text(r, cat, pol):
    if r.random() < 0.10:                                    # short generic review
        txt = r.choice(SHORT[pol])
    else:
        hinglish = r.random() < 0.20
        pool = CATEGORIES[cat]["aspects"] * 2 + UNIVERSAL
        k = r.choice([1, 2, 2, 3])
        keys = r.sample(pool, k=min(k, len(set(pool))))
        keys = list(dict.fromkeys(keys))
        sents = []
        for i, key in enumerate(keys):
            p = pol
            if pol == "neu":
                p = "pos" if i % 2 == 0 else "neg"
                if len(keys) == 1:
                    p = "neu"
            elif r.random() < 0.12 and len(keys) > 1 and i == len(keys) - 1:
                p = "neg" if pol == "pos" else "pos"       # mixed opinion
            s = _aspect_sentence(r, key, p, hinglish)
            if i > 0 and p != "neu" and sents and r.random() < 0.5:
                s = r.choice(["But ", "However, ", "Also, ", "Bas "]) + s[0].lower() + s[1:]
            sents.append(s)
        opener = r.choice({"pos": OPEN_POS, "neg": OPEN_NEG, "neu": OPEN_NEU}[pol])
        closer = r.choice({"pos": CLOSE_POS, "neg": CLOSE_NEG, "neu": CLOSE_NEU}[pol])
        txt = " ".join(x for x in [opener] + sents + [closer] if x)
        if r.random() < 0.08:                                # genuine but 'spammy-looking' sentence
            h = r.choice(HARD_GENUINE).format(d3=r.randint(100, 999), d4=r.randint(1000, 9999),
                                              oid="OD" + "".join(str(r.randint(0, 9)) for _ in range(12)))
            txt += " " + h
    # surface noise
    x = r.random()
    if x < 0.04 and pol == "neg":
        txt = txt.upper()
    elif x < 0.20:
        txt = txt.lower()
    elif x < 0.30:
        txt = txt.translate(str.maketrans("", "", ".,!"))
    if r.random() < 0.08:                                    # typo: swap two neighbouring letters
        i = r.randrange(1, max(2, len(txt) - 2))
        txt = txt[:i] + txt[i + 1] + txt[i] + txt[i + 2:]
    if r.random() < 0.10:
        txt += r.choice(EMOJIS)
    return txt


def _spam_text(r, cat):
    kind = r.choices(["promo_url", "paid_review", "gibberish", "offtopic", "generic_fake_praise", "hybrid", "subtle_fake"],
                     weights=[16, 12, 8, 8, 14, 14, 28])[0]
    if kind == "promo_url":
        parts = [r.choice(SP_HOOK)]
        if r.random() < 0.8:                                 # ~20% have no link / phone: harder
            parts.append(r.choice([f"Visit {_url(r)}", f"Click {_url(r)}", f"WhatsApp {_phone(r)}", f"Call now {_phone(r)}", f"Telegram {_handle(r)}"]))
        txt = " ".join(parts)
    elif kind == "paid_review":
        txt = r.choice(SP_PAID) + (" " + r.choice([f"WhatsApp {_phone(r)}", f"Telegram {_handle(r)}", ""]) if r.random() < 0.5 else "")
    elif kind == "gibberish":
        if r.random() < 0.5:
            txt = " ".join(r.choice(KEYB)[: r.randint(3, 9)] for _ in range(r.randint(2, 6)))
        else:
            w = r.choice(["good", "nice", "product", "ok", "super"])
            txt = " ".join(r.choice([w, "product", "good", "buy"]) for _ in range(r.randint(5, 12)))
    elif kind == "offtopic":
        txt = r.choice(SP_OFF)
    elif kind == "generic_fake_praise":
        txt = r.choice(SP_FAKE)
    elif kind == "subtle_fake":                              # fake praise written like a genuine review, no link
        base = _genuine_text(r, cat, "pos")
        base = " ".join(base.split(".")[:2]).strip()
        txt = " ".join(x for x in [r.choice(SP_SUBTLE), base, r.choice(SP_SUBTLE)] if x)
    else:                                                    # hybrid = genuine-looking + promo tail
        base = _genuine_text(r, cat, "pos")
        base = " ".join(base.split(".")[:2])
        txt = base.strip() + ". " + r.choice(SP_TAIL).format(url=_url(r), handle=_handle(r), phone=_phone(r))
    if r.random() < 0.15:
        txt = txt.lower()
    return txt, kind


# ------------------------------------------------------------------ main generator
def generate(n=8000, seed=42, spam_rate=0.20):
    r = random.Random(seed)
    nr = np.random.default_rng(seed)
    # products
    products = []
    pid = 1
    for cat, cfg in CATEGORIES.items():
        for _ in range(20):
            lo, hi = cfg["price"]
            price = int(np.exp(nr.uniform(np.log(lo), np.log(hi))) // 10 * 10)
            products.append(dict(product_id=f"P{pid:04d}", category=cat,
                                 product_name=f"{r.choice(cfg['brands'])} {r.choice(cfg['items'])} {r.choice(['', 'Plus', 'Max', 'Edition', 'v2'])}".strip(),
                                 platform=r.choice(cfg["platform"]), price_inr=price,
                                 quality=float(nr.beta(5, 2.2) * 0.58 + 0.30)))     # latent product quality 0.30-0.88
            pid += 1
    rows = []
    start = pd.Timestamp("2026-01-01")
    for i in range(n):
        p = r.choice(products)
        u = nr.beta(1.6, 1.0)                                # volume grows over the year
        date = start + pd.Timedelta(days=int(u * 298))
        if r.random() < spam_rate:
            text, kind = _spam_text(r, p["category"])
            rating = 5 if r.random() < 0.75 else r.randint(1, 4)
            spam = 1
        else:
            q = p["quality"]
            pp, pn = q * 0.78, 1 - q * 0.78 - 0.20
            pol = r.choices(["pos", "neu", "neg"], weights=[pp, 0.20, pn])[0]
            text = _genuine_text(r, p["category"], pol)
            rating = {"pos": r.choices([5, 4], [0.62, 0.38])[0], "neu": r.choices([3, 2, 4], [0.75, 0.125, 0.125])[0],
                      "neg": r.choices([1, 2], [0.55, 0.45])[0]}[pol]
            if r.random() < 0.05:                            # rating noise (real users mis-rate)
                rating = r.randint(1, 5)
            spam, kind = 0, "genuine"
        if r.random() < 0.015:                               # annotation noise (moderators disagree)
            spam = 1 - spam
        sentiment = "Positive" if rating >= 4 else ("Neutral" if rating == 3 else "Negative")
        rows.append(dict(review_id=f"R{i + 1:06d}", product_id=p["product_id"], product_name=p["product_name"],
                         category=p["category"], platform=p["platform"], price_inr=p["price_inr"], rating=rating,
                         review_date=date.strftime("%Y-%m-%d"), review_text=text, sentiment=sentiment,
                         is_spam=spam, spam_type=kind))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate()
    print(df.head(10).to_string())
    print(df.is_spam.mean())
