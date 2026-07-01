"""Cohort labeling guidelines, transcribed verbatim from `Labelling Guide.pdf`.

These are the same instructions the human annotators worked from. We hand them
to the LLM unchanged so the man-vs-machine comparison is apples-to-apples: any
disagreement reflects interpretation, not a different rubric.
"""

GENERAL = """\
GENERAL GUIDELINES
- When a specific model of crane is mentioned, use your knowledge to determine the type of crane.
- Mark a sentence "IN" if it fits the cohort, otherwise "OUT".
- Use "MAYBE" sparingly. We really want to know if you think a sentence is in or out.
- If you think a sentence is in a cohort but aren't completely sure, lean towards "IN".
- Only use "MAYBE" when no sentence in the report is currently marked IN and you think a
  sentence might belong to the cohort but you are not sure."""

# One entry per cohort. `key` matches the label string used in the human
# annotations; `definition` is the cohort-specific rubric from the guide.
COHORTS = {
    "Caused by Wind": """\
COHORT: Caused by Wind
Classify a sentence IN if wind is mentioned in the sentence as affecting the crane or
payload at all. It does NOT need to be the primary cause of the accident.""",

    "Mobile Crane Accident": """\
COHORT: Mobile Crane Accident
Classify a sentence IN if the crane mentioned in the sentence is a MOBILE crane (not a
ground- or building-mounted crane).
ONLY MARK SENTENCES THAT SPECIFY THE USE OF A MOBILE CRANE.
Mobile cranes include crawler, wheel, truck, and locomotive-mounted cranes.
Even if the accident does not involve the crane failing (e.g., a worker fell while washing
the crane), if the crane is a mobile crane the sentence should be classified IN.
If the type of crane is not mentioned, classify OUT.""",

    "Static Crane Accident": """\
COHORT: Static Crane Accident
Classify a sentence IN if the crane mentioned in the sentence is a STATIC crane (ground- or
building-mounted).
ONLY MARK SENTENCES THAT SPECIFY THE USE OF A STATIC CRANE.
Static cranes include tower, luffing jib, gantry, overhead, or other permanently or
semi-permanently mounted cranes.
Even if the accident does not involve the crane failing (e.g., a worker fell while washing
the crane), if the crane is a static crane the sentence should be classified IN.
If the type of crane is not mentioned, classify OUT.""",
}


def system_prompt_doc(cohort: str) -> str:
    """Document-level instruction: one IN/OUT/MAYBE decision for the WHOLE report.

    This mirrors how the human annotators actually worked -- of 1193 (report,
    cohort) pairs in the gold data, 1083 are entirely OUT and 109 entirely IN
    (1 mixed): the annotators made a single judgment per report and applied it
    to every sentence. So we ask the model for the same single judgment.
    """
    return f"""You are a careful safety-report annotator. You decide whether an entire \
OSHA-style crane accident report belongs to a specific cohort.

{GENERAL}

{COHORTS[cohort]}

Read the whole report and make ONE decision for the entire report:
- "IN"    if the report belongs to this cohort.
- "OUT"   if it does not.
- "MAYBE" only if the report might belong but you genuinely cannot tell.

Respond with ONLY a JSON object:
{{"label": "IN"|"OUT"|"MAYBE", "evidence": "<short phrase from the report that decided it, or empty>"}}
Do not add commentary."""


def system_prompt(cohort: str) -> str:
    """Full instruction block for one cohort: general rules + cohort rubric + output contract."""
    return f"""You are a careful safety-report annotator. You label each sentence of an OSHA-style \
crane accident report as belonging to a specific cohort or not.

{GENERAL}

{COHORTS[cohort]}

You will be given the full report as a numbered list of sentences, for context. Decide a
value for EVERY sentence index shown.

Respond with ONLY a JSON object of the form:
{{"labels": [{{"index": <int>, "value": "IN"|"OUT"|"MAYBE"}}, ...]}}
Include exactly one entry for every sentence index in the report. Do not add commentary."""
