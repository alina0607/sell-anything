# Data

Small, hand-curated files live here. Large datasets are downloaded into `raw/` and
`processed/`, which are git-ignored.

| File | Description |
|---|---|
| `items.tsv` | All 345 Quick, Draw! categories with a `buy` / `risky` / `refuse` label and the reason for it |
| `traits.json` | Buyer concern tags, trait archetypes, budget levels and names ([spec](../../docs/buyer-ai.md) §2.2) |
| `item_schema.json` | Attributes, answers, value factors and preference signals for all 259 sellable (`buy` and `risky`) items (§2.3). **Generated**: edit `sellanything_ml/buyers/catalog/` and run `python -m sellanything_ml.buyers.catalog.build` |
| `item_splits.json` | Which items are held out of buyer-model training (15% of the sellable items and of `refuse`; items relabelled in the 2026-10-04 review kept the side they were on) |
| `props.json` | Things that can stand next to an item in a buyer's imagined scene: tags, zone, size, which item groups they fit (`buyers/scenes.py`) |
| `cat_buyer_cards.json` | Buyer cards (what each persona cares about in a cat) from the cat sales, used to set up buyers in the game. **Generated** by `python -m sellanything_ml.buyers.sales_sft` |

### `items.tsv` columns

| Column | Meaning |
|---|---|
| `en` | Quick, Draw! category name (exact match with `categories.txt`) |
| `label` | `buy` — the buyers ask questions and make offers; `risky` — dangerous or black-market: buyers whose traits rule it out refuse at once, the rest negotiate; `refuse` — nobody can buy it |
| `group` | Finer category, e.g. `electronics`, `food`, `wildlife`, `body_part` |
| `note` | Why the item has its label, e.g. "bought new only", "retired police cars are auctioned" |
