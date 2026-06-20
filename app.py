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
- Hedging language throughout: "may," "tend to," "could be related to," "appears to," "It has been reported that," "the findings suggest"
- Scholarly third-person voice; no first person; no "we"
- Defines a concept, then illustrates with "For example,"
- Closes paragraphs with measured synthesis using "Therefore," or "It would seem that" or "Such findings suggest" — NOT with absolute claims
- Medium-long sentences with multiple clauses linked by commas
- Avoids markdown bold within narrative prose
═══════════════════════════════════════════════

╔═══════════════════════════════════════════════╗
║ HARD RULE: NO ASSERTIVE OR ABSOLUTE PHRASING  ║
╚═══════════════════════════════════════════════╝
You MUST avoid all forms of definitive, absolute, or assertive claims. The narrative must remain measured, tentative, and hedged at every turn.

PROHIBITED PHRASES (do NOT use any of these):
- "This proves..." / "It proves..." / "Proves that..."
- "Clearly demonstrates..." / "Clearly shows..."
- "Definitively..." / "Definitely..." / "Undoubtedly..." / "Without doubt..."
- "It is certain that..." / "Certainly..."
- "Must..." (in claims about what the field/policy must do)
- "Always..." / "Never..." / "All studies..."
- "The evidence shows..." (use "The evidence suggests..." instead)
- "This establishes..." / "This confirms..."
- "Strongly indicates..." (use "appears to indicate..." instead)
- "It is now clear that..." (use "Such findings suggest..." instead)

REQUIRED HEDGING (use these patterns instead):
- "The findings suggest that..." / "The evidence appears to indicate..."
- "It would seem that..." / "There is some indication that..."
- "The available data point toward..." / "These observations may reflect..."
- "Such patterns could be interpreted as..." / "One possible reading of..."
- "Within the limits of the included studies..." / "On the basis of this corpus..."
- "Tentatively..." / "Provisionally..." / "Cautiously..."

Even in the Conclusion, claims must be framed tentatively. Replace "To conclude, it is now clear that..." with "To conclude, the synthesis tentatively suggests that..." or "On balance, the corpus appears to support..."
"""
    return f"""You are a PhD-level synthesis analyst writing in a specific author's voice for a scoping review on: {c['research_topic']}.

Theoretical anchor: {c['theoretical_anchor']}
{style_section}
TASK
You will receive (1) pre-computed statistical tables about a corpus, and (2) a list of included studies. Your job is to produce ONLY the NARRATIVE TEXT for each numbered section described below. Do NOT regenerate the tables — they are inserted programmatically. Do NOT add markdown headers (the section headings are inserted by the system).

Produce narrative for NINE sections, separated by the literal marker `===SECTION===` on its own line.

RESULTS SECTIONS (sections 1–5, factual interpretation, 100–180 words each)

1. CORPUS OVERVIEW — Interpret what the corpus size, year range, database spread, and tier mix tentatively suggest about the maturity and shape of the evidence base. Hedge throughout.

2. TIER DISTRIBUTION AND EVIDENCE WEIGHT — Interpret which tiers appear to carry the most weight, which seem under-represented, and what this may imply for the strength of the evidence. Refer to specific tier names explicitly.

3. KIRKPATRICK OUTCOME CEILING — Interpret the highest outcome level reached across the intervention studies and what this may mean for policy claims that could or could not be tentatively supported.

4. GEOGRAPHIC AND CONTEXTUAL VARIATION — Discuss how the spread (or absence) of geographic variation could affect generalisability and the application of the theoretical anchor.

5. CROSS-TIER PATTERNS — Identify 2-3 patterns that appear across tiers (e.g., how Tier 3 readiness findings may relate to Tier 4 intervention outcomes, how Tier 1 frameworks may be reflected — or absent — in Tier 4 interventions). Make explicit cross-references between tier numbers so the Discussion can build on them.

DISCUSSION SECTIONS (sections 6–8, building toward an argument, 200–300 words each)

6. COMPARISON WITH AVAILABLE LITERATURE — Position the findings against what comparable scoping reviews, position papers, and frameworks in the wider field have reported. Note convergences and divergences. Where the corpus appears to extend or contradict prior reviews, say so tentatively. Reference the theoretical anchor explicitly. If specific comparator works are not visible in the corpus, frame the comparison at the level of general patterns reported elsewhere in the discipline.

7. INTEGRATION ACROSS TIERS — Build a sustained argument that explicitly links the tier-level findings from the Results section. For example, frame Tier 3 (readiness) and Tier 4 (intervention) findings as complementary or contradictory in light of the Tier 1 (framework). Use phrases like "When read in conjunction with...", "Taken together, Tiers 2 and 3 appear to suggest...", "Such a pattern may be interpreted in light of the theoretical anchor...". This section must reference specific tier numbers and pull the threads together.

8. ARGUMENT AND IMPLICATIONS — Develop the strongest defensible argument the corpus appears to support, while remaining hedged throughout. Distinguish what the corpus may tentatively support at the level of policy/accreditation standards from what it does not. Anticipate counter-positions and address them measuredly. Close this section with a clear (but hedged) claim about what the synthesis appears to contribute.

CONCLUSION (section 9, 120–180 words)

9. CONCLUSION — A closing paragraph framed measuredly. Open with phrases such as "To conclude, the synthesis tentatively suggests that...", "On balance, the corpus appears to indicate...", or "Within the limits of the included studies, the present review provisionally points toward...". Recap the central argument from Section 8 in hedged form, restate the most important research gap, and gesture toward future work.

CRITICAL RULES
- Output narrative ONLY. No markdown headers, no tables, no bullet lists.
- Use the literal marker `===SECTION===` between sections — exactly nine sections.
- Mimic the writing style sample precisely.
- Reference specific tier names and Kirkpatrick levels explicitly in Sections 5, 7, and 8.
- Avoid every form of assertive phrasing listed above. Hedge throughout.
- Do not invent studies or findings beyond what is in the corpus list.
- Do not fabricate citations to specific works that are not in the corpus."""


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


def _short_authors(authors):
    """Convert 'Smith J, Jones K et al.' to 'Smith et al.' or 'Smith and Jones' if 2 authors."""
    if not authors:
        return "—"
    a = authors.strip()
    # If already abbreviated with et al
    if "et al" in a.lower():
        first = a.split(",")[0].strip().split(" ")[0]
        return f"{first} et al."
    parts = [p.strip() for p in a.split(",") if p.strip()]
    if len(parts) == 1:
        return parts[0].split(" ")[0]
    elif len(parts) == 2:
        return f"{parts[0].split(' ')[0]} & {parts[1].split(' ')[0]}"
    else:
        return f"{parts[0].split(' ')[0]} et al."


def _clean_field(v, max_len=80):
    """Clean a field value for table display — handle ?, None, empty, and truncate."""
    if v is None:
        return "—"
    s = str(v).strip().lstrip("?").strip()
    if not s:
        return "—"
    if len(s) > max_len:
        return s[:max_len - 1].rstrip() + "…"
    return s


def build_overview_table_df(corpus, config):
    """Build a pandas DataFrame of included studies for Streamlit display.
    Sorted by tier order then year ascending."""
    if not corpus:
        return None

    tier_order = {t: i for i, t in enumerate(config["tiers"])}
    tier_order[""] = 999  # untiered at end

    def sort_key(c):
        return (tier_order.get(c.get("tier", ""), 999), c.get("year", ""))

    sorted_corpus = sorted(corpus, key=sort_key)

    rows = []
    for i, c in enumerate(sorted_corpus, 1):
        e = c.get("extraction") or {}
        rows.append({
            "#": i,
            "Tier": c.get("tier", "—") or "—",
            "Study": f"{_short_authors(c.get('authors',''))} ({c.get('year','')})",
            "Country/Setting": _clean_field(e.get("country") or e.get("setting"), 40),
            "Design (Sample)": (
                _clean_field(e.get("design"), 50)
                + (f"; n={_clean_field(e.get('sample'), 20)}" if e.get("sample") else "")
            ),
            "AI Tool": _clean_field(e.get("aiTool"), 40),
            "Educational Outcome": _clean_field(e.get("educationalOutcome"), 70),
            "Key Finding": _clean_field(e.get("keyFinding"), 100),
            "Limitation": _clean_field(e.get("mainLimitation"), 70),
            "Kirkpatrick": c.get("kp", "N/A"),
            "PMID/DOI": c.get("pmid", ""),
        })
    return pd.DataFrame(rows)


def build_overview_table_md(corpus, config):
    """Build a markdown overview table grouped by tier (manuscript style)."""
    if not corpus:
        return ""

    tier_order = {t: i for i, t in enumerate(config["tiers"])}
    grouped = {}
    for c in corpus:
        tier = c.get("tier", "") or "(Untiered)"
        grouped.setdefault(tier, []).append(c)

    # Order tiers per config, untiered last
    ordered_tiers = list(config["tiers"]) + [t for t in grouped if t not in config["tiers"]]

    md = []
    counter = 1
    for tier in ordered_tiers:
        studies = grouped.get(tier, [])
        if not studies:
            continue
        studies_sorted = sorted(studies, key=lambda c: c.get("year", ""))
        md.append(f"\n### {tier} (n = {len(studies)})\n")
        md.append("| # | Study | Country | Design (Sample) | AI Tool | Educational Outcome | Key Finding | Kirkpatrick |")
        md.append("|---|---|---|---|---|---|---|---|")
        for c in studies_sorted:
            e = c.get("extraction") or {}
            authors_short = _short_authors(c.get("authors", ""))
            study_cell = f"{authors_short} ({c.get('year','')})"
            country = _clean_field(e.get("country") or e.get("setting"), 30).replace("|", "/")
            design = _clean_field(e.get("design"), 40).replace("|", "/")
            sample = _clean_field(e.get("sample"), 20).replace("|", "/")
            design_cell = f"{design}" + (f"; n={sample}" if sample != "—" else "")
            ai_tool = _clean_field(e.get("aiTool"), 30).replace("|", "/")
            outcome = _clean_field(e.get("educationalOutcome"), 60).replace("|", "/")
            finding = _clean_field(e.get("keyFinding"), 80).replace("|", "/")
            kp = c.get("kp", "N/A")
            md.append(f"| {counter} | {study_cell} | {country} | {design_cell} | {ai_tool} | {outcome} | {finding} | {kp} |")
            counter += 1
    return "\n".join(md)


def categorise_exclusion(rationale):
    """Categorise an exclusion rationale into a standard reason group.
    Returns a short reason label suitable for the PRISMA flowchart."""
    if not rationale:
        return "Other / unspecified"
    r = rationale.lower()
    # Order matters — more specific patterns first
    if any(t in r for t in ["clinical accuracy", "diagnostic accuracy", "ai accuracy"]):
        return "Clinical AI accuracy (no educational outcome)"
    if any(t in r for t in ["llm benchmark", "mcq", "multiple choice", "benchmark"]):
        return "LLM/MCQ benchmark (no curricular implications)"
    if any(t in r for t in ["editorial", "commentary", "letter", "non-peer", "opinion"]):
        return "Editorial / commentary / non-peer-reviewed"
    if any(t in r for t in ["pilot", "n < 30", "small sample", "single class"]):
        return "Small pilot (no institutional scope)"
    if any(t in r for t in ["kap", "knowledge attitude practice", "without curricular"]):
        return "KAP without curricular recommendations"
    if any(t in r for t in ["tangential", "not substantive", "mention only", "passing"]):
        return "Tangential AI mention"
    if any(t in r for t in ["language", "non-english"]):
        return "Non-English / language barrier"
    if any(t in r for t in ["duplicate", "already included"]):
        return "Duplicate record"
    if "criterion" in r or "criteria" in r:
        return "Did not meet inclusion criteria"
    return "Other / unspecified"


def compute_prisma_numbers(results, corpus):
    """Compute PRISMA-ScR numbers from screening results and final corpus."""
    if not results:
        return None

    # Identification — by database
    by_db = {}
    for r in results:
        by_db[r["db"]] = by_db.get(r["db"], 0) + 1

    # Total records identified (sum across DBs before dedup is implicit;
    # but `results` is already deduplicated. We'll report the deduped total
    # as "records after duplicates removed").
    n_after_dedup = len(results)

    # Screening — pending = not yet decided
    n_pending = sum(1 for r in results if r["decision"] == "Pending")
    n_screened = n_after_dedup - n_pending
    n_excluded_screening = sum(1 for r in results if r["decision"] == "Exclude")
    n_maybe = sum(1 for r in results if r["decision"] == "Maybe")
    n_included_screening = sum(1 for r in results if r["decision"] == "Include")

    # Exclusion reason breakdown (from rationales on Exclude items)
    exclusion_reasons = {}
    for r in results:
        if r["decision"] == "Exclude":
            reason = categorise_exclusion(r.get("rationale", ""))
            exclusion_reasons[reason] = exclusion_reasons.get(reason, 0) + 1

    # Final corpus
    n_corpus = len(corpus)

    return {
        "by_db": by_db,
        "n_after_dedup": n_after_dedup,
        "n_screened": n_screened,
        "n_pending": n_pending,
        "n_excluded_screening": n_excluded_screening,
        "n_maybe": n_maybe,
        "n_included_screening": n_included_screening,
        "n_corpus": n_corpus,
        "exclusion_reasons": exclusion_reasons,
    }


def build_prisma_flowchart(prisma):
    """Build a PRISMA-ScR style Mermaid flowchart.

    Layout (top to bottom):
      Identification: Records from each database
      Deduplication: Records after duplicates removed
      Screening: Records screened → Records excluded (with reason breakdown)
      Eligibility: Records assessed for eligibility → Records uncertain (Maybe)
      Included: Studies in final corpus
    """
    if not prisma:
        return "flowchart TD\n    A[No data yet]"

    # Identification block — one node per database
    db_lines = []
    db_total = 0
    for i, (db, n) in enumerate(sorted(prisma["by_db"].items(), key=lambda x: -x[1])):
        db_lines.append(f'    DB{i}["{db}<br/>n = {n:,}"]:::ident')
        db_total += n

    db_to_dedup = "\n".join([f"    DB{i} --> Dedup" for i in range(len(prisma["by_db"]))])

    # Exclusion reason node — list reasons with counts
    exclusion_lines = []
    if prisma["exclusion_reasons"]:
        sorted_reasons = sorted(prisma["exclusion_reasons"].items(), key=lambda x: -x[1])
        for reason, count in sorted_reasons[:6]:  # top 6 to keep diagram readable
            # Mermaid node text — escape pipes and quotes
            safe_reason = reason.replace("|", "/").replace('"', "'")
            exclusion_lines.append(f"• {safe_reason}: n = {count}")
        if len(sorted_reasons) > 6:
            other = sum(c for _, c in sorted_reasons[6:])
            exclusion_lines.append(f"• Other: n = {other}")
    exclusion_text = "<br/>".join(exclusion_lines) if exclusion_lines else "(reasons not categorised)"

    # Build diagram
    chart = f"""flowchart TD
    subgraph IDENT [" Identification "]
{chr(10).join(db_lines)}
    end

    Dedup["Records after duplicates removed<br/>n = {prisma['n_after_dedup']:,}"]:::stage

    Screened["Records screened<br/>n = {prisma['n_screened']:,}"]:::stage

    Excluded["Records excluded at screening<br/>n = {prisma['n_excluded_screening']:,}<br/><br/>Reasons:<br/>{exclusion_text}"]:::excluded

    Eligible["Records assessed for eligibility<br/>n = {prisma['n_included_screening'] + prisma['n_maybe']:,}"]:::stage

    Uncertain["Records uncertain / Maybe<br/>n = {prisma['n_maybe']:,}<br/>(deferred for full-text review)"]:::excluded

    Included["Studies included in synthesis<br/>n = {prisma['n_corpus']:,}"]:::included

{db_to_dedup}
    Dedup --> Screened
    Screened --> Excluded
    Screened --> Eligible
    Eligible --> Uncertain
    Eligible --> Included

    classDef ident fill:#E6F1FB,stroke:#185FA5,color:#0C447C
    classDef stage fill:#F1EFE8,stroke:#5F5E5A,color:#2C2C2A
    classDef excluded fill:#FAECE7,stroke:#993C1D,color:#712B13
    classDef included fill:#E1F5EE,stroke:#0F6E56,color:#085041
"""
    return chart


def build_mermaid_flowchart(stats, config):
    """Legacy conceptual flowchart — kept for backward compatibility.
    The synthesis tab now uses build_prisma_flowchart instead."""
    topic = config["research_topic"][:60]
    top_tiers = sorted(
        [(t, c) for t, c in stats["tier_counts"].items() if c > 0],
        key=lambda x: -x[1]
    )[:5]
    tier_nodes = "\n".join([f'    T{i}["{t} (n={c})"]' for i, (t, c) in enumerate(top_tiers)])
    tier_links = "\n".join([f"    Anchor --> T{i}" for i in range(len(top_tiers))])
    has_l3 = stats["kp_counts"].get("L3 Behaviour", 0) > 0
    has_l4 = stats["kp_counts"].get("L4 Results", 0) > 0
    if has_l4:
        ceiling = "L4 Results reached"
    elif has_l3:
        ceiling = "L3 Behaviour reached"
    else:
        ceiling = "L1/L2 only"
    tier_to_imp = "\n".join([f"    T{i} --> Ceiling" for i in range(len(top_tiers))])
    return f"""flowchart TD
    Anchor["Theoretical anchor:<br/>{topic}"]
{tier_nodes}
    Ceiling["Evidence ceiling:<br/>{ceiling}"]
    Policy["Policy implications"]
{tier_links}
{tier_to_imp}
    Ceiling --> Policy"""


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
    text, err = call_claude(build_synth_system_prompt(), user_msg, max_tokens=6000)

    if err:
        return None, err

    return {"narrative": text, "stats": stats}, None


# ─────────────────────────────────────────────────────────────────
# WORD DOCUMENT EXPORT
# ─────────────────────────────────────────────────────────────────

def build_word_document(corpus, stats, narrative, prisma, config):
    """Build a .docx document with full synthesis content.
    Returns BytesIO object suitable for Streamlit download_button.
    """
    from docx import Document
    from docx.shared import Pt, Cm, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_ALIGN_VERTICAL
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    from io import BytesIO

    doc = Document()

    # ─── Set default font ───
    style = doc.styles['Normal']
    style.font.name = 'Calibri'
    style.font.size = Pt(11)

    # Margins
    for section in doc.sections:
        section.top_margin = Cm(2.0)
        section.bottom_margin = Cm(2.0)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2.5)

    # ─── Title ───
    title = doc.add_heading(f"Synthesis: {config['research_topic']}", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta_run = meta.add_run(f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}  ·  n = {stats['n']} studies  ·  Year range {stats['year_range']}")
    meta_run.italic = True
    meta_run.font.size = Pt(10)
    meta_run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

    doc.add_paragraph()  # spacer

    # ─── PRISMA-ScR Numbers ───
    if prisma:
        doc.add_heading("PRISMA-ScR Flow Summary", level=1)
        p = doc.add_paragraph()
        p.add_run("Records identified across databases: ").bold = True
        p.add_run(", ".join([f"{db} (n = {n:,})" for db, n in sorted(prisma['by_db'].items(), key=lambda x: -x[1])]))

        p = doc.add_paragraph()
        p.add_run("Records after duplicates removed: ").bold = True
        p.add_run(f"n = {prisma['n_after_dedup']:,}")

        p = doc.add_paragraph()
        p.add_run("Records screened: ").bold = True
        p.add_run(f"n = {prisma['n_screened']:,}")

        p = doc.add_paragraph()
        p.add_run("Records excluded at screening: ").bold = True
        p.add_run(f"n = {prisma['n_excluded_screening']:,}")

        if prisma['exclusion_reasons']:
            doc.add_paragraph("Exclusion reasons:", style='Intense Quote')
            for reason, count in sorted(prisma['exclusion_reasons'].items(), key=lambda x: -x[1]):
                bullet = doc.add_paragraph(style='List Bullet')
                bullet.add_run(f"{reason}: n = {count}")

        p = doc.add_paragraph()
        p.add_run("Records uncertain (Maybe / deferred): ").bold = True
        p.add_run(f"n = {prisma['n_maybe']:,}")

        p = doc.add_paragraph()
        p.add_run("Studies included in synthesis: ").bold = True
        run = p.add_run(f"n = {prisma['n_corpus']:,}")
        run.bold = True
        run.font.color.rgb = RGBColor(0x0F, 0x6E, 0x56)

        doc.add_paragraph()

    # ─── Overview Table (Table 1) ───
    doc.add_heading("Table 1. Overview of Included Studies", level=1)
    overview_df = build_overview_table_df(corpus, config)
    if overview_df is not None and not overview_df.empty:
        # Use a compact column subset for Word to avoid overflow
        word_cols = ["#", "Tier", "Study", "Country/Setting", "Design (Sample)", "AI Tool", "Educational Outcome", "Key Finding", "Kirkpatrick"]
        cols = [c for c in word_cols if c in overview_df.columns]
        table = doc.add_table(rows=1, cols=len(cols))
        table.style = 'Light Grid Accent 1'
        table.autofit = True

        # Header row
        hdr_cells = table.rows[0].cells
        for i, col in enumerate(cols):
            cell = hdr_cells[i]
            cell.text = ""
            run = cell.paragraphs[0].add_run(col)
            run.bold = True
            run.font.size = Pt(9)

        # Body rows
        for _, row in overview_df.iterrows():
            cells = table.add_row().cells
            for i, col in enumerate(cols):
                val = str(row[col]) if pd.notna(row[col]) else "—"
                cells[i].text = ""
                run = cells[i].paragraphs[0].add_run(val)
                run.font.size = Pt(8)

        # Tighten column widths approximately
        widths_cm = {
            "#": 0.6, "Tier": 1.8, "Study": 2.2, "Country/Setting": 1.8,
            "Design (Sample)": 2.5, "AI Tool": 1.8, "Educational Outcome": 2.5,
            "Key Finding": 3.5, "Kirkpatrick": 1.4,
        }
        for i, col in enumerate(cols):
            w = widths_cm.get(col, 2.0)
            for cell in table.columns[i].cells:
                cell.width = Cm(w)

    doc.add_page_break()

    # ─── Statistical Tables Section ───
    doc.add_heading("Statistical Tables", level=1)

    # Tier distribution table
    doc.add_heading("Table 2. Tier Distribution", level=2)
    tier_table = doc.add_table(rows=1, cols=3)
    tier_table.style = 'Light Grid Accent 1'
    hdr = tier_table.rows[0].cells
    hdr[0].text = "Tier"
    hdr[1].text = "n"
    hdr[2].text = "% of corpus"
    for cell in hdr:
        for run in cell.paragraphs[0].runs:
            run.bold = True
    for tier, count in stats["tier_counts"].items():
        if count == 0:
            continue
        pct = round(100 * count / stats["n"], 1) if stats["n"] else 0
        row = tier_table.add_row().cells
        row[0].text = tier
        row[1].text = str(count)
        row[2].text = f"{pct}%"

    doc.add_paragraph()

    # Kirkpatrick distribution
    doc.add_heading("Table 3. Kirkpatrick Level Distribution", level=2)
    kp_table = doc.add_table(rows=1, cols=3)
    kp_table.style = 'Light Grid Accent 1'
    hdr = kp_table.rows[0].cells
    hdr[0].text = "Kirkpatrick level"
    hdr[1].text = "n"
    hdr[2].text = "% of corpus"
    for cell in hdr:
        for run in cell.paragraphs[0].runs:
            run.bold = True
    for kp, count in stats["kp_counts"].items():
        if count == 0:
            continue
        pct = round(100 * count / stats["n"], 1) if stats["n"] else 0
        row = kp_table.add_row().cells
        row[0].text = kp
        row[1].text = str(count)
        row[2].text = f"{pct}%"

    doc.add_paragraph()

    # Geographic distribution if available
    if stats.get("geo_counts"):
        doc.add_heading("Table 4. Geographic Distribution", level=2)
        geo_table = doc.add_table(rows=1, cols=2)
        geo_table.style = 'Light Grid Accent 1'
        hdr = geo_table.rows[0].cells
        hdr[0].text = "Country / setting"
        hdr[1].text = "n"
        for cell in hdr:
            for run in cell.paragraphs[0].runs:
                run.bold = True
        for country, count in sorted(stats["geo_counts"].items(), key=lambda x: -x[1])[:15]:
            row = geo_table.add_row().cells
            row[0].text = country
            row[1].text = str(count)
        doc.add_paragraph()

    doc.add_page_break()

    # ─── Narrative Sections ───
    sections = [s.strip() for s in narrative.split("===SECTION===")]
    while len(sections) < 9:
        sections.append("(Section narrative not generated.)")

    section_titles = [
        ("Results", None),
        (None, "1. Corpus Overview"),
        (None, "2. Tier Distribution and Evidence Weight"),
        (None, "3. Kirkpatrick Outcome Ceiling"),
        (None, "4. Geographic and Contextual Variation"),
        (None, "5. Cross-Tier Patterns"),
        ("Discussion", None),
        (None, "6. Comparison with Available Literature"),
        (None, "7. Integration Across Tiers"),
        (None, "8. Argument and Implications"),
        ("Conclusion", None),
        (None, "9. Conclusion"),
    ]

    section_idx = 0
    for parent, sub in section_titles:
        if parent:
            doc.add_heading(parent, level=1)
        if sub:
            doc.add_heading(sub, level=2)
            if section_idx < len(sections):
                # Split narrative into paragraphs
                content = sections[section_idx]
                for para in content.split("\n\n"):
                    para = para.strip()
                    if para:
                        p = doc.add_paragraph(para)
                        p.paragraph_format.space_after = Pt(8)
                        for run in p.runs:
                            run.font.size = Pt(11)
                section_idx += 1

    # Save to BytesIO
    bio = BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio


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
                    with st.spinner("Computing statistics and generating narrative (this may take 60–90s)..."):
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
                    "5. Cross-Tier Patterns",
                    "6. Comparison with Available Literature",
                    "7. Integration Across Tiers",
                    "8. Argument and Implications",
                    "9. Conclusion",
                ]

                # Compute PRISMA numbers for the markdown export
                prisma = compute_prisma_numbers(st.session_state.results, st.session_state.corpus)
                prisma_md = ""
                if prisma:
                    prisma_md = (
                        "## PRISMA-ScR Flow\n\n"
                        f"- Records identified across databases: "
                        + ", ".join([f"{db} n={n:,}" for db, n in sorted(prisma['by_db'].items(), key=lambda x: -x[1])])
                        + f"\n- Records after duplicates removed: n = {prisma['n_after_dedup']:,}\n"
                        f"- Records screened: n = {prisma['n_screened']:,}\n"
                        f"- Records excluded at screening: n = {prisma['n_excluded_screening']:,}\n"
                    )
                    if prisma['exclusion_reasons']:
                        prisma_md += "  - Reasons:\n"
                        for reason, count in sorted(prisma['exclusion_reasons'].items(), key=lambda x: -x[1]):
                            prisma_md += f"    - {reason}: n = {count}\n"
                    prisma_md += (
                        f"- Records uncertain (Maybe / deferred): n = {prisma['n_maybe']:,}\n"
                        f"- Studies included in synthesis: n = {prisma['n_corpus']:,}\n\n"
                    )

                md_parts = [f"# Synthesis: {st.session_state.config['research_topic']}\n"]
                md_parts.append(f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} · n={stats['n']} studies_\n\n")
                if prisma_md:
                    md_parts.append(prisma_md)

                # Overview table (Table 1)
                overview_md = build_overview_table_md(st.session_state.corpus, st.session_state.config)
                if overview_md:
                    md_parts.append("## Table 1. Overview of Included Studies\n")
                    md_parts.append(overview_md + "\n\n")

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

                # ─── Word document download ───
                try:
                    word_bio = build_word_document(
                        st.session_state.corpus,
                        stats,
                        narrative,
                        prisma,
                        st.session_state.config,
                    )
                    st.download_button(
                        "📄 Download (.docx)",
                        data=word_bio,
                        file_name=f"synthesis_{datetime.now().strftime('%Y%m%d_%H%M')}.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        use_container_width=True,
                    )
                except ImportError:
                    st.caption("⚠️ Install `python-docx` to enable Word export.")
                except Exception as e:
                    st.caption(f"⚠️ Word export failed: {e}")

        # ─── Render structured synthesis output ────────────────────
        if st.session_state.synth_data:
            stats = st.session_state.synth_data["stats"]
            narrative = st.session_state.synth_data["narrative"]
            sections = [s.strip() for s in narrative.split("===SECTION===")]

            # Pad sections list so missing ones don't crash
            while len(sections) < 9:
                sections.append("(Section narrative not generated — try regenerating.)")

            st.divider()

            # ───────────── PRISMA-ScR FLOWCHART (at the top) ─────────────
            st.markdown("### PRISMA-ScR Flow Diagram")
            st.caption("Records identified, screened, excluded with reasons, and included in the synthesis.")
            prisma = compute_prisma_numbers(st.session_state.results, st.session_state.corpus)
            if prisma:
                prisma_code = build_prisma_flowchart(prisma)
                render_mermaid(prisma_code, height=720)
                with st.expander("PRISMA flow diagram source (Mermaid)", expanded=False):
                    st.code(prisma_code, language="mermaid")
                with st.expander("PRISMA numbers in tabular form", expanded=False):
                    rows = ["| Stage | n |", "|---|---|"]
                    for db, n in sorted(prisma["by_db"].items(), key=lambda x: -x[1]):
                        rows.append(f"| Records identified — {db} | {n:,} |")
                    rows.append(f"| Records after duplicates removed | {prisma['n_after_dedup']:,} |")
                    rows.append(f"| Records screened | {prisma['n_screened']:,} |")
                    rows.append(f"| Records excluded at screening | {prisma['n_excluded_screening']:,} |")
                    for reason, count in sorted(prisma["exclusion_reasons"].items(), key=lambda x: -x[1]):
                        rows.append(f"|  &nbsp;&nbsp;– {reason} | {count} |")
                    rows.append(f"| Records uncertain (Maybe) | {prisma['n_maybe']:,} |")
                    rows.append(f"| Studies included | {prisma['n_corpus']:,} |")
                    st.markdown("\n".join(rows))
            else:
                st.info("Run a search and screen some results to populate the PRISMA flowchart.")

            st.divider()

            # ──────────────────── TABLE 1: OVERVIEW OF INCLUDED STUDIES ────────────────────
            st.markdown("### Table 1. Overview of Included Studies")
            st.caption(f"All {stats['n']} included studies grouped by tier, ordered by year ascending. Empty cells indicate the field was not extracted or not reported.")
            overview_df = build_overview_table_df(st.session_state.corpus, st.session_state.config)
            if overview_df is not None and not overview_df.empty:
                st.dataframe(
                    overview_df,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "#": st.column_config.NumberColumn(width="small"),
                        "Tier": st.column_config.TextColumn(width="medium"),
                        "Study": st.column_config.TextColumn(width="small"),
                        "Country/Setting": st.column_config.TextColumn(width="small"),
                        "Design (Sample)": st.column_config.TextColumn(width="medium"),
                        "AI Tool": st.column_config.TextColumn(width="small"),
                        "Educational Outcome": st.column_config.TextColumn(width="medium"),
                        "Key Finding": st.column_config.TextColumn(width="large"),
                        "Limitation": st.column_config.TextColumn(width="medium"),
                        "Kirkpatrick": st.column_config.TextColumn(width="small"),
                        "PMID/DOI": st.column_config.TextColumn(width="small"),
                    },
                )
                n_extracted_now = sum(1 for c in st.session_state.corpus if c.get("extraction"))
                if n_extracted_now < stats["n"]:
                    st.caption(f"⚠️ {stats['n'] - n_extracted_now} of {stats['n']} studies have not been AI-extracted. Run AI extraction in the Corpus tab to populate Country, Design, AI Tool, and Key Finding fields.")
            else:
                st.info("No corpus available to display.")

            st.divider()

            # ──────────────────── RESULTS SECTIONS ────────────────────
            st.markdown("## Results")

            # Section 1: Corpus Overview
            st.markdown("### 1. Corpus Overview")
            ov_cols = st.columns(4)
            ov_cols[0].metric("Total studies", stats["n"])
            ov_cols[1].metric("Year range", stats["year_range"])
            ov_cols[2].metric("Databases", len(stats["db_counts"]))
            ov_cols[3].metric("Tiers used", sum(1 for c in stats["tier_counts"].values() if c > 0))
            st.markdown("**Database sourcing**")
            st.markdown(build_db_table(stats))
            st.markdown(sections[0])

            st.divider()

            # Section 2: Tier Distribution
            st.markdown("### 2. Tier Distribution and Evidence Weight")
            st.markdown(build_tier_table(stats, st.session_state.config))
            st.markdown(sections[1])

            st.divider()

            # Section 3: Kirkpatrick Ceiling
            st.markdown("### 3. Kirkpatrick Outcome Ceiling")
            st.markdown(build_kp_table(stats))
            st.markdown(sections[2])

            st.divider()

            # Section 4: Geographic Spread
            st.markdown("### 4. Geographic and Contextual Variation")
            geo_t = build_geo_table(stats)
            if geo_t:
                st.markdown(geo_t)
            else:
                st.info("No geographic data available — run AI extraction on corpus studies for richer geographic detail.")
            st.markdown(sections[3])

            st.divider()

            # Section 5: Cross-Tier Patterns
            st.markdown("### 5. Cross-Tier Patterns")
            st.markdown(sections[4])

            st.divider()

            # ──────────────────── DISCUSSION SECTIONS ────────────────────
            st.markdown("## Discussion")

            # Section 6: Comparison with Available Literature
            st.markdown("### 6. Comparison with Available Literature")
            st.markdown(sections[5])

            st.divider()

            # Section 7: Integration Across Tiers
            st.markdown("### 7. Integration Across Tiers")
            st.markdown(sections[6])

            st.divider()

            # Section 8: Argument and Implications
            st.markdown("### 8. Argument and Implications")
            st.markdown(sections[7])

            st.divider()

            # Section 9: Conclusion
            st.markdown("## Conclusion")
            st.markdown(sections[8])
