"""What a buyer should know about cats, as topics with the facts behind them and where those facts come from.

The training text is written by the teacher model from these facts, each fact several ways (`FORMS`): the
encyclopaedia direction, the reverse direction, questions and answers, a checklist, and what a buyer thinks when a
seller mentions it (the direction the game needs). Allen-Zhu & Li ("Physics of Language Models 3.1", 2023) found
knowledge is only extractable when the same fact appears in many forms during pre-training; Berglund et al. ("The
Reversal Curse", 2023) found a model trained on one direction does not learn the other, so every direction the model
must use is written out.

Topics with dialogue=False appear only in the knowledge text, never in buyer dialogues: whether buyers still ask about
them measures how much knowledge transfers into behaviour on its own.

Facts marked secondary come from summaries of the guidelines, not the guidelines themselves; check them against the
originals before the full run.
"""

from __future__ import annotations

from typing import Any

FORMS = {
    "encyclopedia": "an encyclopaedia paragraph stating these facts",
    "reverse": "a paragraph in which every sentence starts from the sign, situation or consequence and only then "
               "says which cats it concerns, in natural English (for example 'Drinking and urinating more can be a "
               "sign of chronic kidney disease, which is common in senior cats', or 'A household whose cats do not "
               "fight has little to fear from an FIV-positive cat'), the opposite order to an encyclopaedia entry",
    "qa": "four short question-and-answer pairs a new cat owner might ask about this",
    "checklist": "a checklist of what to ask or check before adopting such a cat, each item with a one-line reason",
    "buyer_thought": "three short inner monologues of different buyers at a cat sale who hear the seller mention "
                     "something related and decide what to ask next and why",
    "shuffled": "the encyclopaedia paragraph again with the facts in a different order and different wording",
}

TOPICS: dict[str, dict[str, Any]] = {
    "kitten": {
        "title": "Kittens (birth to 1 year)",
        "facts": [
            "Core vaccines (FVRCP) start at 6 to 8 weeks with a booster every 2 to 4 weeks until about 16 to 18 "
            "weeks (sources differ: 16 to 20 weeks).",
            "Rabies vaccination follows local law and the vaccine label, commonly from 12 weeks.",
            "FeLV vaccination is recommended for kittens, after a test.",
            "Full protection comes 7 to 10 days after the last kitten vaccine; until then a kitten should avoid "
            "unfamiliar cats.",
            "The sensitive socialisation period is about 2 to 7 weeks of age; kittens adopted later can still be "
            "socialised with gentle daily handling.",
            "Veterinary consensus (Feline Fix by Five) is to spay or neuter by 5 months; spaying before the first heat "
            "lowers the risk of mammary cancer.",
        ],
        "sources": ["2021 AAHA/AAFP Feline Life Stage Guidelines", "AAHA/AAFP Feline Vaccination Guidelines "
                    "(secondary summaries)", "Feline Fix by Five task force", "dvm360 / VCA on socialisation"],
        "dialogue": True,
    },
    "young_adult": {
        "title": "Young adult cats (1 to 6 years)",
        "facts": ["Personality is settled by young adulthood.",
                  "Young adults are usually the healthiest and easiest stage to care for; yearly check-ups suffice."],
        "sources": ["2021 AAHA/AAFP Feline Life Stage Guidelines"],
        "dialogue": True,
    },
    "mature_adult": {
        "title": "Mature adult cats (7 to 10 years)",
        "facts": ["From maturity, regular health checks are recommended to catch age-related disease early."],
        "sources": ["2021 AAHA/AAFP Feline Life Stage Guidelines"],
        "dialogue": True,
    },
    "senior": {
        "title": "Senior cats (over 10 years)",
        "facts": [
            "Many cats begin to show age-related changes by about ten years of age; ageing itself is not a disease "
            "and some older cats stay perfectly healthy.",
            "Chronic kidney disease is common in senior cats; signs include drinking and urinating more and lethargy.",
            "Hyperthyroidism is a common hormonal disorder of older cats and can lead to high blood pressure and heart "
            "or kidney problems if untreated.",
            "Dental disease (gum disease, tooth resorption) is common in older cats.",
            "Arthritis is widespread: one study found radiographic signs in about 90% of cats over 12; signs are "
            "reluctance to jump or climb and urinating outside the litter box.",
            "A buyer of a senior cat should ask about recent blood tests, medication or special diets, and mobility.",
        ],
        "sources": ["2021 AAHA/AAFP Feline Life Stage Guidelines", "Cornell Feline Health Center",
                    "Vetstreet senior-cat articles (secondary)"],
        "dialogue": True,
    },
    "dogs": {
        "title": "Cats and dogs",
        "facts": ["Whether a cat has lived with dogs before matters for a home with dogs.",
                  "Cats and dogs are introduced slowly, with the cat having a dog-free retreat."],
        "sources": ["general veterinary behaviour guidance (to be cited)"],
        "dialogue": True,
    },
    "other_cats": {
        "title": "Cats with other cats, FIV and FeLV",
        "facts": [
            "A new cat is usually tested for FIV and FeLV before meeting resident cats.",
            "FIV infects only cats: it cannot pass to people, dogs or other animals.",
            "FIV spreads mainly through deep bite wounds; sharing bowls or grooming each other is not an efficient "
            "route. FeLV spreads more easily: grooming, shared bowls and litter boxes, nursing, fighting and mating.",
            "If the cats of a household do not fight, an FIV-positive cat poses little risk to the others.",
            "If the cats may fight, bringing an FIV-positive cat into a home with FIV-negative cats is a moderate risk.",
            "FIV-positive cats should live indoors with routine veterinary care, because their immune system is weaker.",
            "Kittens can test FIV positive because of their mother's antibodies; a retest at about 6 months is often "
            "advised.",
            "Many cats with FIV or FeLV live reasonably normal lives, but the result affects living with other cats.",
        ],
        "sources": ["Cornell Feline Health Center, Feline Immunodeficiency Virus", "San Francisco SPCA FIV handout",
                    "Nashville Metro Animal Care FIV handout", "Vetstreet / Alley Cat Allies / IDEXX (secondary)"],
        "dialogue": True,
    },
    "children": {
        "title": "Cats and children",
        "facts": ["Whether a cat has met children, how it reacts to noise and handling, and whether it scratches "
                  "when annoyed all matter for a family with young kids."],
        "sources": ["general veterinary behaviour guidance (to be cited)"],
        "dialogue": True,
    },
    "breed_hcm": {
        "title": "Heart disease in Ragdolls and Maine Coons",
        "facts": ["Ragdolls and Maine Coons are predisposed to hypertrophic cardiomyopathy (HCM), linked to a "
                  "MYBPC3 mutation; breeders can screen for it."],
        "sources": ["Kinship / The Wildest breed-health articles (secondary)"],
        "dialogue": True,
    },
    "breed_pkd": {
        "title": "Polycystic kidney disease in Persians",
        "facts": ["Polycystic kidney disease (PKD) was first identified in Persians and is common in Persian-related "
                  "breeds; a genetic test and ultrasound can detect it."],
        "sources": ["PMC6656814 (PKD1 mutation prevalence)", "Genomia PKD test"],
        "dialogue": False,
    },
    "breed_fold": {
        "title": "Joint disease in Scottish Folds",
        "facts": ["The gene that folds a Scottish Fold's ears also causes osteochondrodysplasia, a progressive "
                  "arthritis; many vets regard breeding for folded ears as a welfare problem."],
        "sources": ["Kinship / The Wildest (secondary); Australian study of heterozygous Folds"],
        "dialogue": False,
    },
    "allergies": {
        "title": "Cat allergies",
        "facts": [
            "No cat is truly hypoallergenic: every cat makes Fel d 1, the main allergen.",
            "Fel d 1 levels can differ up to 100-fold between individual cats; hairless and curly-coated breeds may "
            "make somewhat less; males make more, and neutering before puberty lowers it.",
            "An allergic buyer should spend time with the particular cat before adopting.",
        ],
        "sources": ["The Conversation (2022)", "Purina (secondary)"],
        "dialogue": True,
    },
    "outdoors": {
        "title": "Indoor and outdoor cats",
        "facts": ["Outdoor access exposes a cat to traffic, fights and infectious disease; a cat used to roaming may "
                  "struggle to adapt to an indoor-only home."],
        "sources": ["general veterinary guidance (to be cited)"],
        "dialogue": True,
    },
    "care": {
        "title": "Everyday care",
        "facts": ["Long-haired cats need regular brushing.", "Some cats are picky about litter; scratching posts "
                  "protect furniture.", "Vaccines, check-ups and senior-cat care are recurring costs.",
                  "A cat's coat colour or pattern (tabby, calico, orange, black) says nothing about coat length, "
                  "shedding or grooming needs."],
        "sources": ["Cornell Feline Health Center (to be cited per fact)"],
        "dialogue": True,
    },
}

HELD_OUT = [t for t, v in TOPICS.items() if not v["dialogue"]]


def knowledge_prompt(topic: str, form: str) -> str:
    """The teacher's instruction for one topic in one form, with the facts it must stay true to."""
    t = TOPICS[topic]
    facts = "\n".join(f"- {f}" for f in t["facts"])
    return (f"Topic: {t['title']}.\nFacts (stay strictly true to these; do not add medical claims beyond them):\n"
            f"{facts}\n\nWrite {FORMS[form]}. Use only the facts above: do not add symptoms, numbers or medical claims "
            "that are not listed. Plain English, no headings, no markdown. Always name the subject in full (for example "
            "'senior cats', 'Persian cats'), never just 'they'.")
