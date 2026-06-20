"""
Universal Literature Research Tool
Search PubMed, Europe PMC, OpenAlex + AI screening via Anthropic API
Configurable inclusion criteria, tiers, and Kirkpatrick levels for any research field
"""

import streamlit as st
import requests
import json
import time
import pandas as pd
from anthropic import Anthropic
from datetime import datetime
from urllib.parse import quote

st.set_page_config(
    page_title="Literature Research Tool",
    page_icon="📚",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────────
# CONFIG: User-editable presets
# ─────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "research_topic": "Artificial Intelligence in Dental Education and Accreditation",
    "theoretical_anchor": "the accreditation dilemma — structural tension between formal accreditation requirements and dental schools' institutional capacity to implement them (DiMaggio & Powell isomorphism; Scott's three pillars — regulative, normative, cultural-cognitive; loose coupling; Meyer & Rowan myth and ceremony)",
    "inclusion_criteria": """1. Proposes or evaluates an AI competency/curriculum framework for the field
2. Empirical study of practitioner or learner AI readiness with explicit curricular/policy recommendations (n ≥ 30)
3. Educational intervention integrating AI into curriculum with institutional scope
4. Studies accreditation alignment, quality assurance, or educational policy regarding AI
5. Foundational position paper cited by professional/accrediting bodies""",
    "exclusion_criteria": """A. Pure clinical AI diagnostic accuracy study with no educational outcome
B. LLM benchmark/MCQ accuracy test without curricular implications
C. Single-class pilot n < 30 with no institutional integration discussion
D. KAP survey without explicit curricular recommendations
E. Editorial, commentary, letter, or non-peer-reviewed material
F. Tangential AI mention (AI not a substantive component of the study)""",
    "tiers": [
        "Tier 1 – Framework",
        "Tier 2 – Needs Evidence",
        "Tier 3 – Practitioner Readiness",
        "Tier 4 – Intervention",
        "Tier 5 – Measurement",
        "Tier 6 – Review",
    ],
    "tier_descriptions": """Tier 1 – Framework: proposes/validates a competency or curriculum framework
Tier 2 – Needs Evidence: empirical evidence on learner/practitioner educational needs
Tier 3 – Practitioner Readiness: workforce preparedness, awareness, attitudes
Tier 4 – Intervention: tests an educational intervention with institutional scope
Tier 5 – Measurement: capability benchmarked against educational standards
Tier 6 – Review: scoping/systematic review or bibliometric analysis""",
    "kp_levels": ["N/A", "L1 Reaction", "L2 Learning", "L3 Behaviour", "L4 Results"],
    "kp_description": """N/A – framework, review, or non-intervention study
L1 – Reaction: attitudes, satisfaction, perceived value
L2 – Learning: knowledge/skill acquisition (objective or self-reported)
L3 – Behaviour: observed change in practice
L4 – Results: patient or institutional outcomes (Yardley & Dornan 2012)""",
    "search_presets": [
        ("AI + Dental + Accreditation", '("artificial intelligence" OR "machine learning" OR "large language model" OR "ChatGPT") AND ("dental education" OR "dental curriculum") AND ("accreditation" OR "quality assurance")'),
        ("Educator readiness", '("artificial intelligence") AND ("dental faculty" OR "dental educator") AND ("readiness" OR "attitude" OR "perception")'),
        ("AI curriculum frameworks", '("artificial intelligence") AND ("dental education") AND ("curriculum" OR "competency" OR "framework")'),
        ("GDC / UK standards", '("artificial intelligence") AND ("dental education") AND ("General Dental Council" OR "GDC" OR "United Kingdom")'),
        ("NCAAA / Saudi Arabia", '("dental education") AND ("Saudi Arabia" OR "NCAAA") AND ("quality" OR "accreditation")'),
    ],
}

# ─────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────

def init_state():
    defaults = {
        "config": DEFAULT_CONFIG.copy(),
        "results": [],
        "corpus": [],
        "search_status": "",
        "db_status": {},
        "screening_id": None,
        "synth_text": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─────────────────────────────────────────────────────────────────
# DATABASE SEARCH FUNCTIONS
# ─────────────────────────────────────────────────────────────────

def search_pubmed(query, date_from, date_to, max_results):
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    try:
        s = requests.get(
            f"{base}/esearch.fcgi",
            params={
                "db": "pubmed",
                "term": query,
                "mindate": date_from,
                "maxdate": date_to,
                "retmax": max_results,
                "retmode": "json",
            },
            timeout=15,
        )
        s.raise_for_status()
        sd = s.json()
        ids = sd.get("esearchresult", {}).get("idlist", [])
        if not ids:
            return [], 0
        f = requests.get(
            f"{base}/esummary.fcgi",
            params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"},
            timeout=15,
        )
        f.raise_for_status()
        fd = f.json()
        items = []
        for pmid in ids:
            d = fd.get("result", {}).get(pmid)
            if not d:
                continue
            title = (d.get("title") or "No title").replace("<", " ").replace(">", " ")
            authors = ", ".join([a.get("name", "") for a in d.get("authors", [])[:3]])
            if len(d.get("authors", [])) > 3:
                authors += " et al."
            items.append({
                "id": f"pm_{pmid}",
                "pmid": pmid,
                "title": title,
                "authors": authors,
                "journal": d.get("fulljournalname") or d.get("source", ""),
                "year": (d.get("pubdate") or "").split(" ")[0],
                "db": "PubMed",
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "abstract": "",
                "decision": "Pending",
                "tier": "",
                "kp": "N/A",
                "rationale": "",
                "confidence": "",
                "extraction": None,
            })
        return items, int(sd.get("esearchresult", {}).get("count", len(items)))
    except Exception as e:
        st.warning(f"PubMed error: {e}")
        return [], 0


def search_epmc(query, date_from, date_to, max_results):
    try:
        r = requests.get(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            params={
                "query": query,
                "fromYear": date_from,
                "toYear": date_to,
                "pageSize": max_results,
                "format": "json",
                "resultType": "core",
            },
            timeout=20,
        )
        r.raise_for_status()
        d = r.json()
        items = []
        for a in d.get("resultList", {}).get("result", []):
            authors_list = a.get("authorList", {}).get("author", [])
            authors = ", ".join([(u.get("fullName") or u.get("lastName") or "") for u in authors_list[:3]])
            if len(authors_list) > 3:
                authors += " et al."
            pmid = a.get("pmid") or a.get("id", "")
            items.append({
                "id": f"ep_{pmid}",
                "pmid": pmid,
                "title": (a.get("title") or "No title").replace("<", " ").replace(">", " "),
                "authors": authors,
                "journal": a.get("journalTitle", ""),
                "year": str(a.get("pubYear", "")),
                "db": "Europe PMC",
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{a.get('pmid')}/" if a.get("pmid") else f"https://europepmc.org/article/{a.get('source','MED')}/{a.get('id')}",
                "abstract": a.get("abstractText", ""),
                "decision": "Pending",
                "tier": "",
                "kp": "N/A",
                "rationale": "",
                "confidence": "",
                "extraction": None,
            })
        return items, d.get("hitCount", len(items))
    except Exception as e:
        st.warning(f"Europe PMC error: {e}")
        return [], 0


def search_openalex(query, date_from, date_to, max_results):
    try:
        filter_str = f"default.search:{quote(query)},from_publication_date:{date_from}-01-01,to_publication_date:{date_to}-12-31"
        r = requests.get(
            f"https://api.openalex.org/works",
            params={
                "filter": filter_str,
                "per-page": max_results,
                "select": "id,title,authorships,publication_year,primary_location,doi",
                "mailto": "research.tool@example.com",
            },
            timeout=20,
        )
        r.raise_for_status()
        d = r.json()
        items = []
        for w in d.get("results", []):
            wid = w["id"].split("/")[-1]
            authors_list = w.get("authorships", [])
            authors = ", ".join([(a.get("author", {}).get("display_name") or "") for a in authors_list[:3] if a.get("author")])
            if len(authors_list) > 3:
                authors += " et al."
            doi = w.get("doi", "")
            items.append({
                "id": f"oa_{wid}",
                "pmid": doi.replace("https://doi.org/", "") if doi else wid,
                "title": w.get("title") or "No title",
                "authors": authors,
                "journal": (w.get("primary_location") or {}).get("source", {}).get("display_name", "") if (w.get("primary_location") or {}).get("source") else "",
                "year": str(w.get("publication_year", "")),
                "db": "OpenAlex",
                "url": doi if doi else f"https://openalex.org/{wid}",
                "abstract": "",
                "decision": "Pending",
                "tier": "",
                "kp": "N/A",
                "rationale": "",
                "confidence": "",
                "extraction": None,
            })
        return items, d.get("meta", {}).get("count", len(items))
    except Exception as e:
        st.warning(f"OpenAlex error: {e}")
        return [], 0


def dedup_results(items):
    seen = set()
    out = []
    for it in items:
        k = "".join(c for c in (it["title"] or "").lower() if c.isalnum())[:50]
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(it)
    return out


# ─────────────────────────────────────────────────────────────────
# CLAUDE AI FUNCTIONS
# ─────────────────────────────────────────────────────────────────

def get_anthropic_client():
    """Read API key from secrets or environment."""
    key = None
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        pass
    if not key:
        import os
        key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    return Anthropic(api_key=key)


def build_screen_system_prompt():
    c = st.session_state.config
    tiers_list = " | ".join(c["tiers"])
    kp_list = " | ".join(c["kp_levels"])
    return f"""You are a systematic review screener for a scoping review on: {c['research_topic']}.

Theoretical anchor: {c['theoretical_anchor']}

INCLUSION CRITERIA (must meet at least one):
{c['inclusion_criteria']}

EXCLUSION CRITERIA (any one excludes):
{c['exclusion_criteria']}

TIER DEFINITIONS:
{c['tier_descriptions']}

KIRKPATRICK LEVELS:
{c['kp_description']}

Respond ONLY as valid JSON with no markdown, no preamble:
{{"decision":"Include|Exclude|Maybe","tier":"<one of: {tiers_list}>","kp":"<one of: {kp_list}>","rationale":"One sentence citing which criterion drove the decision.","confidence":"high|medium|low"}}"""


def build_extract_system_prompt():
    c = st.session_state.config
    return f"""You are a data extractor for a scoping review on: {c['research_topic']}.
Based on the article title, journal, tier and year, infer the most plausible study characteristics.
For any field where inference is uncertain, prefix with "?".
Respond ONLY as valid JSON, no markdown:
{{"country":"","setting":"","design":"","sample":"","aiTool":"","educationalOutcome":"","keyFinding":"one sentence","mainLimitation":"one sentence","relevance":"direct|partial|indirect","notes":""}}"""


def build_synth_system_prompt():
    c = st.session_state.config
    return f"""You are a PhD-level synthesis analyst for a scoping review on: {c['research_topic']}.

Theoretical anchor: {c['theoretical_anchor']}

Write a structured synthesis covering:
1. Tier distribution and what it reveals about the evidential base
2. Kirkpatrick level ceiling and what it means for policy
3. Geographic spread and contextual variation
4. Key research gaps
5. What this corpus can and cannot defensibly support at policy level

PhD voice, ~400-500 words, flowing prose without markdown headers."""


def call_claude(system_prompt, user_message, max_tokens=1000):
    client = get_anthropic_client()
    if not client:
        return None, "No Anthropic API key configured. Add ANTHROPIC_API_KEY to Streamlit secrets or environment."
    try:
        msg = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return msg.content[0].text, None
    except Exception as e:
        return None, str(e)


def screen_article(article):
    user_msg = f"""Title: {article['title']}
Authors: {article['authors']}
Journal: {article['journal']} ({article['year']})
Database: {article['db']}"""
    if article.get("abstract"):
        user_msg += f"\nAbstract: {article['abstract'][:800]}"
    text, err = call_claude(build_screen_system_prompt(), user_msg)
    if err:
        return None, err
    try:
        cleaned = text.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned), None
    except Exception as e:
        return None, f"Parse error: {e}\nRaw: {text[:200]}"


def extract_article(article):
    user_msg = f"""Title: {article['title']}
Authors: {article['authors']}
Journal: {article['journal']} ({article['year']})
Database: {article['db']}
Tier: {article.get('tier','?')}
Kirkpatrick: {article.get('kp','N/A')}
Screening rationale: {article.get('rationale','')}"""
    if article.get("abstract"):
        user_msg += f"\nAbstract: {article['abstract'][:800]}"
    text, err = call_claude(build_extract_system_prompt(), user_msg)
    if err:
        return None, err
    try:
        cleaned = text.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned), None
    except Exception as e:
        return None, f"Parse error: {e}\nRaw: {text[:200]}"


def synthesise_corpus(corpus):
    if not corpus:
        return None, "Corpus is empty."
    dbs = ", ".join(set(c["db"] for c in corpus))
    listing = "\n".join([
        f"- {c['authors']} ({c['year']}) [{c.get('tier','?')}] [{c.get('kp','N/A')}] [{c['db']}]: {c['title']}"
        + (f" | Finding: {c['extraction']['keyFinding']}" if c.get("extraction") and c["extraction"].get("keyFinding") else "")
        for c in corpus
    ])
    user_msg = f"Corpus (n={len(corpus)}) from databases: {dbs}\n\n{listing}"
    return call_claude(build_synth_system_prompt(), user_msg, max_tokens=2000)


# ─────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────

DECISION_COLORS = {
    "Include": "#0F6E56",
    "Maybe": "#854F0B",
    "Exclude": "#993C1D",
    "Pending": "#888780",
}
DECISION_BG = {
    "Include": "#E1F5EE",
    "Maybe": "#FAEEDA",
    "Exclude": "#FAECE7",
    "Pending": "#F1EFE8",
}
DB_COLORS = {
    "PubMed": ("#E6F1FB", "#0C447C"),
    "Europe PMC": ("#EAF3DE", "#27500A"),
    "OpenAlex": ("#EEEDFE", "#3C3489"),
}


def badge(text, bg, color, font_size="11px"):
    return f'<span style="display:inline-block;background:{bg};color:{color};padding:2px 8px;border-radius:6px;font-size:{font_size};font-weight:500;margin-right:4px">{text}</span>'


def db_badge(db):
    bg, col = DB_COLORS.get(db, ("#F1EFE8", "#444441"))
    return badge(db, bg, col)


# ─────────────────────────────────────────────────────────────────
# HEADER & SIDEBAR
# ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
.stApp { font-family: -apple-system, BlinkMacSystemFont, sans-serif; }
.block-container { padding-top: 1.5rem; max-width: 1200px; }
</style>
""", unsafe_allow_html=True)

st.title("📚 Literature Research Tool")
st.caption("Multi-database search · AI-powered screening · structured extraction · synthesis")

# API key status
client = get_anthropic_client()
if not client:
    st.error("⚠️ No Anthropic API key found. Add `ANTHROPIC_API_KEY` to Streamlit secrets (`.streamlit/secrets.toml`) or environment variables.")
    st.info("AI screening, extraction and synthesis features require the API key. Database search works without it.")

# Sidebar: Config
with st.sidebar:
    st.header("⚙️ Configuration")
    with st.expander("Research topic & criteria", expanded=False):
        st.session_state.config["research_topic"] = st.text_input(
            "Research topic",
            value=st.session_state.config["research_topic"],
        )
        st.session_state.config["theoretical_anchor"] = st.text_area(
            "Theoretical anchor (used in screening & synthesis)",
            value=st.session_state.config["theoretical_anchor"],
            height=100,
        )
        st.session_state.config["inclusion_criteria"] = st.text_area(
            "Inclusion criteria",
            value=st.session_state.config["inclusion_criteria"],
            height=150,
        )
        st.session_state.config["exclusion_criteria"] = st.text_area(
            "Exclusion criteria",
            value=st.session_state.config["exclusion_criteria"],
            height=150,
        )

    with st.expander("Tiers & Kirkpatrick", expanded=False):
        tiers_text = st.text_area(
            "Tiers (one per line)",
            value="\n".join(st.session_state.config["tiers"]),
            height=120,
        )
        st.session_state.config["tiers"] = [t.strip() for t in tiers_text.split("\n") if t.strip()]
        st.session_state.config["tier_descriptions"] = st.text_area(
            "Tier descriptions",
            value=st.session_state.config["tier_descriptions"],
            height=120,
        )
        kp_text = st.text_area(
            "Kirkpatrick levels (one per line)",
            value="\n".join(st.session_state.config["kp_levels"]),
            height=120,
        )
        st.session_state.config["kp_levels"] = [k.strip() for k in kp_text.split("\n") if k.strip()]
        st.session_state.config["kp_description"] = st.text_area(
            "Kirkpatrick descriptions",
            value=st.session_state.config["kp_description"],
            height=120,
        )

    st.divider()
    if st.button("Reset to defaults"):
        st.session_state.config = DEFAULT_CONFIG.copy()
        st.rerun()

    st.divider()
    st.markdown("**Session stats**")
    st.metric("Results", len(st.session_state.results))
    st.metric("Corpus", len(st.session_state.corpus))

# ─────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────

tab_search, tab_screen, tab_corpus, tab_synth = st.tabs(["🔍 Search", "✓ Screen", "📂 Corpus", "📝 Synthesis"])

# ─── SEARCH TAB ────────────────────────────────────────────────
with tab_search:
    st.subheader("Database search")

    preset_cols = st.columns(min(5, len(st.session_state.config["search_presets"])))
    for i, (label, query) in enumerate(st.session_state.config["search_presets"]):
        with preset_cols[i % len(preset_cols)]:
            if st.button(label, key=f"preset_{i}", use_container_width=True):
                st.session_state.pending_query = query
                st.rerun()

    default_q = st.session_state.get("pending_query", st.session_state.config["search_presets"][0][1])
    query = st.text_area("Search query", value=default_q, height=100, key="search_query_input")

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        date_from = st.number_input("From year", min_value=1990, max_value=2030, value=2018)
    with c2:
        date_to = st.number_input("To year", min_value=1990, max_value=2030, value=2026)
    with c3:
        max_per_db = st.slider("Max per DB", 5, 100, 20, step=5)
    with c4:
        st.write("")
        st.write("")
        search_btn = st.button("🔍 Search all databases", type="primary", use_container_width=True)

    selected_dbs = st.multiselect(
        "Databases to search",
        ["PubMed", "Europe PMC", "OpenAlex"],
        default=["PubMed", "Europe PMC", "OpenAlex"],
    )

    if search_btn and query.strip():
        progress = st.progress(0, text="Searching...")
        all_results = []
        status_lines = []

        if "PubMed" in selected_dbs:
            progress.progress(15, text="Querying PubMed...")
            items, total = search_pubmed(query, date_from, date_to, max_per_db)
            all_results.extend(items)
            status_lines.append(f"PubMed: {len(items)} of {total:,} total")
            progress.progress(40, text="Done PubMed")

        if "Europe PMC" in selected_dbs:
            progress.progress(50, text="Querying Europe PMC...")
            items, total = search_epmc(query, date_from, date_to, max_per_db)
            all_results.extend(items)
            status_lines.append(f"Europe PMC: {len(items)} of {total:,} total")
            progress.progress(70, text="Done Europe PMC")

        if "OpenAlex" in selected_dbs:
            progress.progress(80, text="Querying OpenAlex...")
            items, total = search_openalex(query, date_from, date_to, max_per_db)
            all_results.extend(items)
            status_lines.append(f"OpenAlex: {len(items)} of {total:,} total")
            progress.progress(95, text="Done OpenAlex")

        deduped = dedup_results(all_results)
        for i, r in enumerate(deduped):
            r["num"] = i + 1
        st.session_state.results = deduped
        st.session_state.search_status = f"**{len(deduped)} unique results** after deduplication"
        st.session_state.db_status = status_lines
        progress.progress(100, text=f"Done — {len(deduped)} unique results")
        time.sleep(0.5)
        progress.empty()
        st.rerun()

    if st.session_state.search_status:
        st.success(st.session_state.search_status)
        st.write(" · ".join(st.session_state.db_status))

    if st.session_state.results:
        inc = sum(1 for r in st.session_state.results if r["decision"] == "Include")
        exc = sum(1 for r in st.session_state.results if r["decision"] == "Exclude")
        may = sum(1 for r in st.session_state.results if r["decision"] == "Maybe")
        pen = sum(1 for r in st.session_state.results if r["decision"] == "Pending")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total", len(st.session_state.results))
        m2.metric("Include", inc)
        m3.metric("Maybe", may)
        m4.metric("Exclude", exc)
        st.info(f"📋 Switch to the **Screen tab** to apply AI screening against your inclusion criteria.")


# ─── SCREEN TAB ────────────────────────────────────────────────
with tab_screen:
    st.subheader("Screening")

    if not st.session_state.results:
        st.info("Run a search first.")
    else:
        # Filters
        fc1, fc2, fc3, fc4 = st.columns([1, 1, 2, 2])
        with fc1:
            filter_dec = st.selectbox("Decision filter", ["All", "Pending", "Include", "Maybe", "Exclude"], key="filter_dec")
        with fc2:
            filter_db = st.selectbox("Database filter", ["All", "PubMed", "Europe PMC", "OpenAlex"], key="filter_db")
        with fc3:
            n_pending = sum(1 for r in st.session_state.results if r["decision"] == "Pending")
            if st.button(f"🤖 AI screen all pending ({n_pending})", type="primary", disabled=not client or n_pending == 0):
                if client:
                    progress = st.progress(0, text="Starting AI screening...")
                    pending = [r for r in st.session_state.results if r["decision"] == "Pending"]
                    for i, art in enumerate(pending):
                        progress.progress((i + 1) / len(pending), text=f"Screening {i+1}/{len(pending)}: {art['title'][:60]}...")
                        result, err = screen_article(art)
                        if result:
                            for r in st.session_state.results:
                                if r["id"] == art["id"]:
                                    r["decision"] = result.get("decision", "Maybe")
                                    r["tier"] = result.get("tier", "")
                                    r["kp"] = result.get("kp", "N/A")
                                    r["rationale"] = result.get("rationale", "")
                                    r["confidence"] = result.get("confidence", "")
                                    break
                        else:
                            for r in st.session_state.results:
                                if r["id"] == art["id"]:
                                    r["decision"] = "Maybe"
                                    r["rationale"] = f"AI error: {err}"
                                    break
                        time.sleep(0.4)
                    progress.empty()
                    st.success(f"Screened {len(pending)} articles.")
                    st.rerun()
        with fc4:
            if st.button(f"➕ Add all Includes to corpus", type="secondary"):
                existing = {c["id"] for c in st.session_state.corpus}
                added = 0
                for r in st.session_state.results:
                    if r["decision"] == "Include" and r["id"] not in existing:
                        st.session_state.corpus.append(dict(r))
                        added += 1
                st.success(f"Added {added} studies to corpus.")
                st.rerun()

        # Apply filters
        filtered = st.session_state.results
        if filter_dec != "All":
            filtered = [r for r in filtered if r["decision"] == filter_dec]
        if filter_db != "All":
            filtered = [r for r in filtered if r["db"] == filter_db]

        st.caption(f"Showing {len(filtered)} of {len(st.session_state.results)} results")

        for r in filtered:
            with st.container(border=True):
                col_main, col_btns = st.columns([5, 2])
                with col_main:
                    st.markdown(f"**#{r['num']}** · [{r['title']}]({r['url']})")
                    st.caption(f"{r['authors']} · *{r['journal']}* {r['year']} · {r['db']}")
                with col_btns:
                    dec_cols = st.columns(3)
                    for j, d in enumerate(["Include", "Maybe", "Exclude"]):
                        with dec_cols[j]:
                            current = r["decision"] == d
                            if st.button(d, key=f"dec_{r['id']}_{d}", type="primary" if current else "secondary", use_container_width=True):
                                for rr in st.session_state.results:
                                    if rr["id"] == r["id"]:
                                        rr["decision"] = d
                                        break
                                st.rerun()

                # Tier & Kirkpatrick row
                tk_cols = st.columns([3, 3, 2, 2])
                with tk_cols[0]:
                    tier_opts = [""] + st.session_state.config["tiers"]
                    current_tier_idx = tier_opts.index(r["tier"]) if r["tier"] in tier_opts else 0
                    new_tier = st.selectbox("Tier", tier_opts, index=current_tier_idx, key=f"tier_{r['id']}", label_visibility="collapsed")
                    if new_tier != r["tier"]:
                        for rr in st.session_state.results:
                            if rr["id"] == r["id"]:
                                rr["tier"] = new_tier
                                break
                with tk_cols[1]:
                    kp_opts = st.session_state.config["kp_levels"]
                    current_kp_idx = kp_opts.index(r["kp"]) if r["kp"] in kp_opts else 0
                    new_kp = st.selectbox("KP", kp_opts, index=current_kp_idx, key=f"kp_{r['id']}", label_visibility="collapsed")
                    if new_kp != r["kp"]:
                        for rr in st.session_state.results:
                            if rr["id"] == r["id"]:
                                rr["kp"] = new_kp
                                break
                with tk_cols[2]:
                    if st.button("🤖 AI screen", key=f"aiscr_{r['id']}", disabled=not client, use_container_width=True):
                        with st.spinner("Screening..."):
                            result, err = screen_article(r)
                            if result:
                                for rr in st.session_state.results:
                                    if rr["id"] == r["id"]:
                                        rr["decision"] = result.get("decision", "Maybe")
                                        rr["tier"] = result.get("tier", "")
                                        rr["kp"] = result.get("kp", "N/A")
                                        rr["rationale"] = result.get("rationale", "")
                                        rr["confidence"] = result.get("confidence", "")
                                        break
                                st.rerun()
                            else:
                                st.error(f"Error: {err}")
                with tk_cols[3]:
                    if r["decision"] == "Include":
                        if st.button("➕ Corpus", key=f"corp_{r['id']}", use_container_width=True):
                            if not any(c["id"] == r["id"] for c in st.session_state.corpus):
                                st.session_state.corpus.append(dict(r))
                            st.success("Added")
                            st.rerun()

                if r["rationale"]:
                    conf_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(r.get("confidence", ""), "")
                    st.caption(f"*{conf_emoji} {r['rationale']}*")


# ─── CORPUS TAB ────────────────────────────────────────────────
with tab_corpus:
    st.subheader(f"Corpus — {len(st.session_state.corpus)} studies")

    if not st.session_state.corpus:
        st.info("Mark articles as Include in the Screen tab, then add them to the corpus.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            n_unextracted = sum(1 for c in st.session_state.corpus if not c.get("extraction"))
            if st.button(f"🤖 AI extract all ({n_unextracted} remaining)", type="primary", disabled=not client or n_unextracted == 0):
                if client:
                    progress = st.progress(0, text="Extracting...")
                    to_extract = [c for c in st.session_state.corpus if not c.get("extraction")]
                    for i, art in enumerate(to_extract):
                        progress.progress((i + 1) / len(to_extract), text=f"Extracting {i+1}/{len(to_extract)}")
                        result, err = extract_article(art)
                        for c in st.session_state.corpus:
                            if c["id"] == art["id"]:
                                c["extraction"] = result if result else {"error": err}
                                break
                        time.sleep(0.4)
                    progress.empty()
                    st.success(f"Extracted {len(to_extract)} studies.")
                    st.rerun()
        with c2:
            # CSV export
            rows = []
            for c in st.session_state.corpus:
                e = c.get("extraction") or {}
                rows.append({
                    "PMID/DOI": c["pmid"],
                    "Title": c["title"],
                    "Authors": c["authors"],
                    "Journal": c["journal"],
                    "Year": c["year"],
                    "Database": c["db"],
                    "Decision": c["decision"],
                    "Confidence": c.get("confidence", ""),
                    "Tier": c.get("tier", ""),
                    "Kirkpatrick": c.get("kp", ""),
                    "Screening rationale": c.get("rationale", ""),
                    "Country": e.get("country", ""),
                    "Setting": e.get("setting", ""),
                    "Design": e.get("design", ""),
                    "Sample": e.get("sample", ""),
                    "AI tool": e.get("aiTool", ""),
                    "Educational outcome": e.get("educationalOutcome", ""),
                    "Key finding": e.get("keyFinding", ""),
                    "Main limitation": e.get("mainLimitation", ""),
                    "Relevance": e.get("relevance", ""),
                    "URL": c["url"],
                })
            df = pd.DataFrame(rows)
            st.download_button(
                "📥 Export corpus CSV",
                df.to_csv(index=False),
                file_name=f"corpus_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True,
            )

        for c in st.session_state.corpus:
            with st.container(border=True):
                col_main, col_btn = st.columns([6, 1])
                with col_main:
                    st.markdown(f"**[{c['title']}]({c['url']})**")
                    badges = c["db"]
                    if c.get("tier"):
                        badges += f" · {c['tier']}"
                    if c.get("kp") and c["kp"] != "N/A":
                        badges += f" · {c['kp']}"
                    st.caption(f"{c['authors']} · *{c['journal']}* {c['year']} · {badges}")
                    if c.get("rationale"):
                        st.caption(f"*{c['rationale']}*")
                with col_btn:
                    if st.button("Remove", key=f"rm_{c['id']}", use_container_width=True):
                        st.session_state.corpus = [x for x in st.session_state.corpus if x["id"] != c["id"]]
                        st.rerun()

                if c.get("extraction"):
                    e = c["extraction"]
                    if "error" in e:
                        st.error(f"Extraction error: {e['error']}")
                    else:
                        ec1, ec2 = st.columns(2)
                        with ec1:
                            for k in ["country", "setting", "design", "sample"]:
                                if e.get(k):
                                    st.markdown(f"**{k.capitalize()}:** {e[k]}")
                        with ec2:
                            for k in ["aiTool", "educationalOutcome", "relevance"]:
                                if e.get(k):
                                    label = {"aiTool": "AI tool", "educationalOutcome": "Outcome", "relevance": "Relevance"}[k]
                                    st.markdown(f"**{label}:** {e[k]}")
                        if e.get("keyFinding"):
                            st.markdown(f"**Key finding:** {e['keyFinding']}")
                        if e.get("mainLimitation"):
                            st.markdown(f"**Limitation:** {e['mainLimitation']}")


# ─── SYNTHESIS TAB ────────────────────────────────────────────
with tab_synth:
    st.subheader("Corpus synthesis")
    st.caption(f"Synthesises your corpus (n={len(st.session_state.corpus)}) against your theoretical anchor.")

    if not st.session_state.corpus:
        st.info("Add studies to the corpus first.")
    else:
        if st.button("✨ Generate synthesis", type="primary", disabled=not client):
            if client:
                with st.spinner("Analysing corpus..."):
                    text, err = synthesise_corpus(st.session_state.corpus)
                    if text:
                        st.session_state.synth_text = text
                    else:
                        st.error(f"Error: {err}")

        if st.session_state.synth_text:
            st.markdown(st.session_state.synth_text)
            st.download_button(
                "📥 Download synthesis (markdown)",
                st.session_state.synth_text,
                file_name=f"synthesis_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown",
            )
