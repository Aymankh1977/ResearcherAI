# 📚 Literature Research Tool

A configurable Streamlit app for systematic and scoping reviews. Searches PubMed, Europe PMC, and OpenAlex; uses Claude AI to screen articles against your inclusion/exclusion criteria; extracts structured data; and synthesises findings — all from the same interface.

Works for any research field — defaults are dental education AI, but you fully configure the topic, theoretical anchor, criteria, tiers, and Kirkpatrick levels in the sidebar.

---

## ✨ Features

- **Three database search** in one click — PubMed, Europe PMC, OpenAlex
- **Automatic deduplication** by title across databases
- **AI screening** against your inclusion/exclusion criteria with rationale and confidence
- **Tier classification** (configurable — Tier 1 to Tier 6 by default)
- **Kirkpatrick level** classification (Yardley & Dornan 2012 adaptation)
- **Structured data extraction** per study (country, design, sample, AI tool, key finding, etc.)
- **Synthesis generation** with your theoretical anchor
- **CSV export** of full corpus including all extracted fields
- **Per-session storage** — no database required

---

## 🚀 Quick Start — Streamlit Cloud (recommended)

### 1. Get an Anthropic API key
Sign up at https://console.anthropic.com → Settings → API keys → Create key.
You'll need a small credit (~$5) loaded for screening to work.

### 2. Fork / upload to GitHub
- Create a new public GitHub repo
- Upload `app.py`, `requirements.txt`, `.gitignore`, and `README.md`
- **Never upload `secrets.toml`** — it's in `.gitignore` already

### 3. Deploy on Streamlit Cloud
- Go to https://share.streamlit.io
- Sign in with GitHub
- Click **New app** → select your repo → main branch → `app.py`
- Click **Advanced settings** → **Secrets** → paste:
  ```toml
  ANTHROPIC_API_KEY = "sk-ant-your-key-here"
  ```
- Click **Deploy**

That's it. Your tool is now live at a `*.streamlit.app` URL.

---

## 💻 Local Development

```bash
# 1. Clone or download
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

# 2. Install dependencies (Python 3.10+)
pip install -r requirements.txt

# 3. Add your API key
mkdir -p .streamlit
echo 'ANTHROPIC_API_KEY = "sk-ant-your-key-here"' > .streamlit/secrets.toml

# 4. Run
streamlit run app.py
```

Opens at http://localhost:8501

---

## 🎯 Workflow

1. **Search tab** — pick a preset or enter a query, choose databases and date range, click Search
2. **Screen tab** — click "AI screen all pending" for batch screening, or screen individual articles
3. **Corpus tab** — Includes are added to corpus; "AI extract all" fills structured fields per study
4. **Synthesis tab** — generate a theoretical synthesis of your corpus
5. **Export** CSV any time from the Corpus tab

---

## ⚙️ Configuration

Open the **Configuration** panel in the sidebar to customise:

- **Research topic** — what your review is about
- **Theoretical anchor** — used in screening rationale and synthesis (e.g. institutional theory, conceptual framework)
- **Inclusion criteria** — the rules an article must meet to be included
- **Exclusion criteria** — the rules that automatically exclude
- **Tiers** — how you organise the evidence (one per line)
- **Tier descriptions** — what each tier captures
- **Kirkpatrick levels** — your outcome level definitions (one per line)

Default configuration is calibrated for AI in dental education and the accreditation dilemma framework. Edit freely for any field.

---

## 🔧 Adapting Search Presets for Your Field

Edit the `DEFAULT_CONFIG["search_presets"]` list near the top of `app.py`:

```python
"search_presets": [
    ("My field preset 1", '("term1" OR "term2") AND ("term3")'),
    ("My field preset 2", '("...") AND ("...")'),
],
```

Use PubMed query syntax — works across all three databases.

---

## 💰 API Costs

Approximate costs (Claude Sonnet 4.5):
- Screening 100 articles: ~$0.30
- Extracting 30 studies: ~$0.15
- Generating one synthesis: ~$0.05

A typical scoping review (500 articles screened, 30 in corpus, 1 synthesis) costs roughly $2.

---

## 🆘 Troubleshooting

**"No Anthropic API key found"** — Add `ANTHROPIC_API_KEY` to Streamlit secrets (cloud) or `.streamlit/secrets.toml` (local).

**OpenAlex returns 0 results** — Their search syntax differs slightly. Try simpler queries.

**PubMed rate limit** — Rare with default settings. If hit, wait 60s.

**AI screening returns "Maybe" for everything** — Tighten your inclusion/exclusion criteria in the sidebar config.

**Out of API credit** — Top up at https://console.anthropic.com → Plans & billing.

---

## 📄 License

MIT. Use freely for academic and commercial research.
