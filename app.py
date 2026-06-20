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
    "writing_style_sample": """In dental education, assessment of students is of critical value to improve their performance in clinical settings. The assessment of students' clinical performance includes various stages, where in certain cases, assessors encounter challenges in providing final grades. This study sheds a light on assessment practices in clinical settings and focuses on assessors' modulation of the whole cognitive process. The argument involves discussing critical thinking of assessors before, during, and after the event of assessment. Then, it analyzes a cognitive approach of assessment implied by assessors during students' performance. Further, it proposes a model with step-by-step approach in decision-making along with different factors, which may strongly influence final grades. Four main stages were identified for the purpose of analysis, such as pre-decision, driver, primary decision, and communication stages. Each stage was supported by literary data, along with evidences worth consideration.

Briefly, the four stages explain the flow of information during the cognitive process with conjoint factors. For example, internal and external sources are the main factors to affect a pre-decision (non-task-specific) cognitive stage in clinical settings. The driver stage starts when the assessor is present to judge the performance of specific clinical tasks of assessment. The third stage is the primary decision stage, which begins when the assessor finds or sees (interpret) students' performance according to the defined frame of reference. The resulted primary decision predicts a range of options between 'being sure' and 'uncertainty'. Then, the refinement process of decision grade, the fourth stage, is the moderation of the decision, which is affected by another set of factors, such as legal consequences, community, and patient safety to direct the decision towards grading.

Expertise is believed to be a primary internal factor, which influences clinical reasoning to update assessors' judgment capacity. Clinical reasoning consists of two types: content-dependent and context-dependent. However, in therapeutic and diagnostic reasoning, expertise generally differs among assessors. For example, in psychology, deliberate practice acts as a key to expert performance. It may further benefit clinical reasoning as that experts possess more stringent decisions than early career assessors. In the light of behavioral learning perspective theory, expertise is guided by disciplines' specific knowledge and skills, which in turn affect internal assessors' attitudes, emotions, intentions, and personalities.

Interestingly, one external factor that is found to increase the stringency of assessors' decision is the increased number of candidates during time of assessment. This could be related to the feeling of fatigue or other factors, which may be further investigated in future studies. To conclude, it is now clear that the above-mentioned internal and external factors can influence the cognitive process during the pre-decision stage to initially predict assessors' appraisals of learners' performances.""",
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
        "synth_data": None,
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

        # Get summaries (titles, authors, journal, year)
        f = requests.get(
            f"{base}/esummary.fcgi",
            params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"},
            timeout=15,
        )
        f.raise_for_status()
        fd = f.json()

        # Get abstracts via efetch (separate call - needed for accurate AI screening)
        abstracts = {}
        try:
            ab = requests.get(
                f"{base}/efetch.fcgi",
                params={"db": "pubmed", "id": ",".join(ids), "rettype": "abstract", "retmode": "xml"},
                timeout=20,
            )
            ab.raise_for_status()
            import re
            xml = ab.text
            # Simple regex parse — split by article and extract PMID + abstract
            articles = re.split(r'<PubmedArticle>', xml)
            for art in articles[1:]:
                pmid_m = re.search(r'<PMID[^>]*>(\d+)</PMID>', art)
                abs_m = re.findall(r'<AbstractText[^>]*>(.*?)</AbstractText>', art, re.DOTALL)
                if pmid_m:
                    abstracts[pmid_m.group(1)] = " ".join([re.sub(r'<[^>]+>', '', a).strip() for a in abs_m])[:2000]
        except Exception:
            pass  # Abstracts are nice-to-have, not critical

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
                "abstract": abstracts.get(pmid, ""),
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
        # Europe PMC supports PubMed-style queries but works best with PUB_YEAR filter
        epmc_query = f"({query}) AND (PUB_YEAR:[{date_from} TO {date_to}])"
        r = requests.get(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            params={
                "query": epmc_query,
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
        # OpenAlex works better with the 'search' parameter (full-text) than 'default.search' filter
        # Strip Boolean operators that OpenAlex doesn't handle well
        clean_query = query.replace(' AND ', ' ').replace(' OR ', ' ').replace('(', '').replace(')', '').replace('"', '')
        r = requests.get(
            "https://api.openalex.org/works",
            params={
                "search": clean_query,
                "filter": f"from_publication_date:{date_from}-01-01,to_publication_date:{date_to}-12-31",
                "per-page": max_results,
                "select": "id,title,authorships,publication_year,primary_location,doi,abstract_inverted_index",
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

            # Reconstruct abstract from OpenAlex inverted index
            abstract = ""
            inv = w.get("abstract_inverted_index")
            if inv:
                word_positions = []
                for word, positions in inv.items():
                    for pos in positions:
                        word_positions.append((pos, word))
                word_positions.sort()
                abstract = " ".join([w for _, w in word_positions])[:2000]

            items.append({
                "id": f"oa_{wid}",
                "pmid": doi.replace("https://doi.org/", "") if doi else wid,
                "title": w.get("title") or "No title",
                "authors": authors,
                "journal": (w.get("primary_location") or {}).get("source", {}).get("display_name", "") if (w.get("primary_location") or {}).get("source") else "",
                "year": str(w.get("publication_year", "")),
                "db": "OpenAlex",
                "url": doi if doi else f"https://openalex.org/{wid}",
                "abstract": abstract,
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
    style_section = ""
    if c.get("writing_style_sample", "").strip():
        style_section = f"""

═══════════════════════════════════════════════
WRITING STYLE — MIMIC THIS VOICE EXACTLY
═══════════════════════════════════════════════
The following passages illustrate the author's writing voice. Match this style precisely in every narrative section you produce: sentence rhythm, vocabulary, hedging patterns, paragraph openers, use of connective adverbs (Moreover, Therefore, However, Interestingly, Briefly, In fact, Consequently), academic third-person tone, and the characteristic move of defining a concept then giving an example.

STYLE SAMPLE:
\"\"\"
{c['writing_style_sample']}
\"\"\"

Distinctive features to replicate:
- Opens sentences with connective adverbs: "Moreover,", "Therefore,", "However,", "Interestingly,", "In fact,", "Briefly,", "Consequently,", "Nevertheless,"
- Hedging language: "may," "tend to," "could be related to," "It has been reported that"
- Scholarly third-person voice; no first person; no "we"
- Defines a concept, then illustrates with "For example,"
- Closes paragraphs with synthesis using "Therefore," or "To conclude," or "It is now clear that"
- Medium-long sentences with multiple clauses linked by commas
- Avoids markdown bold within narrative prose
═══════════════════════════════════════════════
"""
    return f"""You are a PhD-level synthesis analyst writing in a specific author's voice for a scoping review on: {c['research_topic']}.

Theoretical anchor: {c['theoretical_anchor']}
{style_section}
TASK
You will receive (1) pre-computed statistical tables about a corpus, and (2) a list of included studies. Your job is to produce ONLY the NARRATIVE TEXT for each numbered section described below. Do NOT regenerate the tables — they are inserted programmatically. Do NOT add markdown headers (the section headings are inserted by the system). Each narrative section should be 100–200 words of flowing prose in the author's voice.

Produce narrative for SEVEN sections, separated by the literal marker `===SECTION===` on its own line:

1. CORPUS OVERVIEW — interpret what the corpus size, year range, database spread, and tier mix reveal about the maturity and shape of the evidence base.

2. TIER DISTRIBUTION — interpret which tiers carry the most weight, which are under-represented, and what this implies for the strength of the evidence in answering the review question.

3. KIRKPATRICK CEILING — interpret the highest outcome level reached across the intervention studies and what this means for policy claims that can or cannot be made.

4. GEOGRAPHIC AND CONTEXTUAL VARIATION — discuss how the spread (or absence) of geographic variation affects generalisability and the application of the theoretical anchor.

5. KEY RESEARCH GAPS — identify 3-4 specific gaps the corpus reveals, anchoring each gap in the theoretical framework.

6. POLICY IMPLICATIONS — distinguish what the corpus can defensibly support at the level of accreditation/regulatory standards from what it cannot, and why.

7. CONCLUSION — a closing paragraph synthesising the review's contribution in the author's voice, opening with "To conclude," or "It is now clear that".

CRITICAL RULES
- Output narrative ONLY. No markdown headers, no tables, no bullet lists.
- Use the literal marker `===SECTION===` between sections.
- Mimic the writing style sample precisely.
- Reference specific tier names, Kirkpatrick levels, and statistics from the data provided.
- Do not invent studies or findings beyond what is in the corpus list."""


def compute_corpus_stats(corpus, config):
    """Compute structured statistics from the corpus for tables."""
    if not corpus:
        return {}

    n = len(corpus)
    years = [int(c["year"]) for c in corpus if c.get("year", "").isdigit()]
    year_range = f"{min(years)}–{max(years)}" if years else "—"

    # Tier distribution
    tier_counts = {}
    for t in config["tiers"]:
        tier_counts[t] = sum(1 for c in corpus if c.get("tier") == t)
    untiered = sum(1 for c in corpus if not c.get("tier"))
    if untiered:
        tier_counts["(Untiered)"] = untiered

    # Kirkpatrick distribution
    kp_counts = {}
    for k in config["kp_levels"]:
        kp_counts[k] = sum(1 for c in corpus if c.get("kp") == k)

    # Database distribution
    db_counts = {}
    for c in corpus:
        db_counts[c["db"]] = db_counts.get(c["db"], 0) + 1

    # Geographic distribution from extractions
    geo_counts = {}
    design_counts = {}
    relevance_counts = {"direct": 0, "partial": 0, "indirect": 0}
    for c in corpus:
        e = c.get("extraction") or {}
        country = (e.get("country") or "").strip().lstrip("?").strip()
        if country:
            geo_counts[country] = geo_counts.get(country, 0) + 1
        design = (e.get("design") or "").strip().lstrip("?").strip()
        if design:
            design_counts[design] = design_counts.get(design, 0) + 1
        rel = (e.get("relevance") or "").strip().lower()
        if rel in relevance_counts:
            relevance_counts[rel] += 1

    return {
        "n": n,
        "year_range": year_range,
        "tier_counts": tier_counts,
        "kp_counts": kp_counts,
        "db_counts": db_counts,
        "geo_counts": geo_counts,
        "design_counts": design_counts,
        "relevance_counts": relevance_counts,
    }


def build_tier_table(stats, config):
    """Render a tier distribution table as markdown + a list of top studies per tier."""
    rows = ["| Tier | n | % of corpus |", "|---|---|---|"]
    total = stats["n"]
    for tier, count in stats["tier_counts"].items():
        if count == 0:
            continue
        pct = round(100 * count / total, 1) if total else 0
        rows.append(f"| {tier} | {count} | {pct}% |")
    return "\n".join(rows)


def build_kp_table(stats):
    rows = ["| Kirkpatrick level | n | % of corpus |", "|---|---|---|"]
    total = stats["n"]
    for kp, count in stats["kp_counts"].items():
        if count == 0:
            continue
        pct = round(100 * count / total, 1) if total else 0
        rows.append(f"| {kp} | {count} | {pct}% |")
    return "\n".join(rows)


def build_geo_table(stats):
    if not stats["geo_counts"]:
        return None
    rows = ["| Country / setting | n |", "|---|---|"]
    sorted_geo = sorted(stats["geo_counts"].items(), key=lambda x: -x[1])
    for country, count in sorted_geo[:15]:
        rows.append(f"| {country} | {count} |")
    return "\n".join(rows)


def build_db_table(stats):
    rows = ["| Database | n studies sourced |", "|---|---|"]
    for db, count in sorted(stats["db_counts"].items(), key=lambda x: -x[1]):
        rows.append(f"| {db} | {count} |")
    return "\n".join(rows)


def build_mermaid_flowchart(stats, config):
    """Build a Mermaid flowchart linking theoretical anchor → evidence → implications."""
    topic = config["research_topic"][:60]
    top_tiers = sorted(
        [(t, c) for t, c in stats["tier_counts"].items() if c > 0],
        key=lambda x: -x[1]
    )[:5]
    tier_nodes = "\n".join([f'    T{i}["{t} (n={c})"]' for i, (t, c) in enumerate(top_tiers)])
    tier_links = "\n".join([f"    Anchor --> T{i}" for i in range(len(top_tiers))])
    # Implications based on KP ceiling
    has_l3 = stats["kp_counts"].get("L3 Behaviour", 0) > 0
    has_l4 = stats["kp_counts"].get("L4 Results", 0) > 0
    if has_l4:
        ceiling = "L4 Results reached<br/>(strongest policy claims possible)"
    elif has_l3:
        ceiling = "L3 Behaviour reached<br/>(practice-level claims defensible)"
    else:
        ceiling = "L1/L2 only<br/>(knowledge-level claims only)"

    tier_to_imp = "\n".join([f"    T{i} --> Ceiling" for i in range(len(top_tiers))])

    return f"""flowchart TD
    Anchor["Theoretical anchor:<br/>{topic}"]
{tier_nodes}
    Ceiling["Evidence ceiling:<br/>{ceiling}"]
    Policy["Policy implications<br/>(see Section 6)"]
{tier_links}
{tier_to_imp}
    Ceiling --> Policy
    classDef anchor fill:#E6F1FB,stroke:#0C447C,color:#0C447C
    classDef tier fill:#E1F5EE,stroke:#085041,color:#085041
    classDef ceiling fill:#FAEEDA,stroke:#633806,color:#633806
    classDef policy fill:#EEEDFE,stroke:#3C3489,color:#3C3489
    class Anchor anchor
    class Ceiling ceiling
    class Policy policy"""


def render_mermaid(diagram_code, height=420):
    """Render a Mermaid diagram inside Streamlit via embedded HTML."""
    html = f"""
    <div style="background:white;border-radius:8px;padding:1rem;">
      <pre class="mermaid" style="background:transparent;">
{diagram_code}
      </pre>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <script>
      mermaid.initialize({{ startOnLoad: true, theme: 'default', securityLevel: 'loose' }});
    </script>
    """
    st.components.v1.html(html, height=height, scrolling=False)


def synthesise_corpus(corpus, config):
    """Generate structured synthesis: pre-computed tables + AI narrative in user's voice."""
    if not corpus:
        return None, "Corpus is empty."

    stats = compute_corpus_stats(corpus, config)

    # Build study listing for narrative context
    listing = "\n".join([
        f"- {c['authors']} ({c['year']}) [{c.get('tier','?')}] [{c.get('kp','N/A')}] [{c['db']}]: {c['title']}"
        + (f" | Country: {c['extraction'].get('country','')}" if c.get("extraction") and c["extraction"].get("country") else "")
        + (f" | Finding: {c['extraction']['keyFinding']}" if c.get("extraction") and c["extraction"].get("keyFinding") else "")
        for c in corpus
    ])

    # Pass pre-computed stats to Claude as context
    stats_summary = f"""PRE-COMPUTED CORPUS STATISTICS (use these exact numbers in your narrative — do not recompute):

Total studies: {stats['n']}
Year range: {stats['year_range']}

Tier distribution:
{chr(10).join([f"  • {t}: {c}" for t, c in stats['tier_counts'].items() if c > 0])}

Kirkpatrick level distribution:
{chr(10).join([f"  • {k}: {c}" for k, c in stats['kp_counts'].items() if c > 0])}

Database sourcing:
{chr(10).join([f"  • {d}: {c}" for d, c in sorted(stats['db_counts'].items(), key=lambda x: -x[1])])}

Geographic distribution (from extractions):
{chr(10).join([f"  • {g}: {c}" for g, c in sorted(stats['geo_counts'].items(), key=lambda x: -x[1])[:10]]) if stats['geo_counts'] else "  • (No extraction data — geographic synthesis will be limited)"}

Study design distribution (from extractions):
{chr(10).join([f"  • {d}: {c}" for d, c in sorted(stats['design_counts'].items(), key=lambda x: -x[1])[:8]]) if stats['design_counts'] else "  • (No extraction data)"}
"""

    user_msg = stats_summary + "\n\nCORPUS LISTING:\n" + listing
    text, err = call_claude(build_synth_system_prompt(), user_msg, max_tokens=4000)

    if err:
        return None, err

    return {"narrative": text, "stats": stats}, None


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

    with st.expander("✍️ Writing style", expanded=False):
        st.caption("Paste 2-3 paragraphs of your own academic writing. Claude will mimic your sentence rhythm, vocabulary, hedging language, and paragraph openers in the synthesis narrative.")
        st.session_state.config["writing_style_sample"] = st.text_area(
            "Writing style sample (your own published work)",
            value=st.session_state.config.get("writing_style_sample", ""),
            height=250,
            help="Best results: paste 300-500 words from your most representative paper. The voice features are extracted automatically.",
        )
        if st.session_state.config.get("writing_style_sample", "").strip():
            words = len(st.session_state.config["writing_style_sample"].split())
            st.caption(f"✓ {words} words loaded · style will be applied to synthesis narrative")
        else:
            st.warning("No style sample set — synthesis will use a generic academic voice.")

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
        max_per_db = st.slider("Max per DB", 10, 200, 50, step=10, help="How many results to fetch from EACH database. 50 per DB = up to 150 total.")
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
    st.subheader("Structured corpus synthesis")
    n_corpus = len(st.session_state.corpus)
    has_style = bool(st.session_state.config.get("writing_style_sample", "").strip())
    n_extracted = sum(1 for c in st.session_state.corpus if c.get("extraction"))

    cap_bits = [f"Corpus n={n_corpus}"]
    cap_bits.append("✓ writing style loaded" if has_style else "⚠ no writing style set")
    cap_bits.append(f"{n_extracted}/{n_corpus} extracted")
    st.caption(" · ".join(cap_bits))

    if n_corpus == 0:
        st.info("Add studies to the corpus first, then optionally run AI extraction for richer geographic/design tables.")
    else:
        c1, c2 = st.columns([2, 1])
        with c1:
            if st.button("✨ Generate structured synthesis", type="primary", disabled=not client):
                if client:
                    with st.spinner("Computing statistics and generating narrative..."):
                        result, err = synthesise_corpus(st.session_state.corpus, st.session_state.config)
                        if result:
                            st.session_state.synth_data = result
                            st.session_state.synth_text = result["narrative"]
                        else:
                            st.error(f"Error: {err}")
        with c2:
            if st.session_state.synth_data:
                # Build downloadable markdown
                stats = st.session_state.synth_data["stats"]
                narrative = st.session_state.synth_data["narrative"]
                sections = [s.strip() for s in narrative.split("===SECTION===")]
                section_titles = [
                    "1. Corpus Overview",
                    "2. Tier Distribution and Evidence Weight",
                    "3. Kirkpatrick Outcome Ceiling",
                    "4. Geographic and Contextual Variation",
                    "5. Key Research Gaps",
                    "6. Policy Implications",
                    "7. Conclusion",
                ]
                md_parts = [f"# Synthesis: {st.session_state.config['research_topic']}\n"]
                md_parts.append(f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} · n={stats['n']} studies_\n\n")
                for i, title in enumerate(section_titles):
                    md_parts.append(f"## {title}\n")
                    # Add table for relevant sections
                    if i == 0:
                        md_parts.append(f"**Total studies:** {stats['n']}  ·  **Year range:** {stats['year_range']}\n\n")
                        md_parts.append("### Database sourcing\n" + build_db_table(stats) + "\n\n")
                    elif i == 1:
                        md_parts.append("### Tier distribution\n" + build_tier_table(stats, st.session_state.config) + "\n\n")
                    elif i == 2:
                        md_parts.append("### Kirkpatrick level distribution\n" + build_kp_table(stats) + "\n\n")
                    elif i == 3:
                        geo_t = build_geo_table(stats)
                        if geo_t:
                            md_parts.append("### Geographic distribution\n" + geo_t + "\n\n")
                    if i < len(sections):
                        md_parts.append(sections[i] + "\n\n")
                full_md = "".join(md_parts)
                st.download_button(
                    "📥 Download (.md)",
                    full_md,
                    file_name=f"synthesis_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

        # ─── Render structured synthesis output ────────────────────
        if st.session_state.synth_data:
            stats = st.session_state.synth_data["stats"]
            narrative = st.session_state.synth_data["narrative"]
            sections = [s.strip() for s in narrative.split("===SECTION===")]

            st.divider()

            # Section 1: Corpus Overview
            st.markdown("### 1. Corpus Overview")
            ov_cols = st.columns(4)
            ov_cols[0].metric("Total studies", stats["n"])
            ov_cols[1].metric("Year range", stats["year_range"])
            ov_cols[2].metric("Databases", len(stats["db_counts"]))
            ov_cols[3].metric("Tiers used", sum(1 for c in stats["tier_counts"].values() if c > 0))
            st.markdown("**Database sourcing**")
            st.markdown(build_db_table(stats))
            if len(sections) > 0 and sections[0]:
                st.markdown(sections[0])

            st.divider()

            # Section 2: Tier Distribution
            st.markdown("### 2. Tier Distribution and Evidence Weight")
            st.markdown(build_tier_table(stats, st.session_state.config))
            if len(sections) > 1 and sections[1]:
                st.markdown(sections[1])

            st.divider()

            # Section 3: Kirkpatrick Ceiling
            st.markdown("### 3. Kirkpatrick Outcome Ceiling")
            st.markdown(build_kp_table(stats))
            if len(sections) > 2 and sections[2]:
                st.markdown(sections[2])

            st.divider()

            # Section 4: Geographic Spread
            st.markdown("### 4. Geographic and Contextual Variation")
            geo_t = build_geo_table(stats)
            if geo_t:
                st.markdown(geo_t)
            else:
                st.info("No geographic data available — run AI extraction on corpus studies for this section.")
            if len(sections) > 3 and sections[3]:
                st.markdown(sections[3])

            st.divider()

            # Synthesis Flowchart
            st.markdown("### Synthesis Flowchart")
            st.caption("Visual map of the theoretical anchor, evidence tiers, and policy ceiling.")
            mermaid_code = build_mermaid_flowchart(stats, st.session_state.config)
            render_mermaid(mermaid_code, height=480)

            with st.expander("Flowchart source (Mermaid)", expanded=False):
                st.code(mermaid_code, language="mermaid")

            st.divider()

            # Section 5: Research Gaps
            st.markdown("### 5. Key Research Gaps")
            if len(sections) > 4 and sections[4]:
                st.markdown(sections[4])

            st.divider()

            # Section 6: Policy Implications
            st.markdown("### 6. Policy Implications")
            if len(sections) > 5 and sections[5]:
                st.markdown(sections[5])

            st.divider()

            # Section 7: Conclusion
            st.markdown("### 7. Conclusion")
            if len(sections) > 6 and sections[6]:
                st.markdown(sections[6])
