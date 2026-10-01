# Data

Small, hand-curated files live here. Large datasets are downloaded into `raw/` and
`processed/`, which are git-ignored.

| File | Description |
|---|---|
| `items.tsv` | All 345 Quick, Draw! categories with a Traditional Chinese name and a `buy` / `refuse` label |

### `items.tsv` columns

| Column | Meaning |
|---|---|
| `en` | Quick, Draw! category name (exact match with `categories.txt`) |
| `zh` | Traditional Chinese name |
| `label` | `buy` — the buyer asks questions and makes an offer; `refuse` — the buyer declines |
| `group` | Finer category, e.g. `electronics`, `food`, `wildlife`, `body_part` |
| `note` | Extra guidance for dialogue generation, e.g. "new only" |
