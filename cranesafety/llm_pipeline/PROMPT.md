# Portable crane-cohort labeling prompt

A single, model-agnostic prompt you can send to **any** LLM (local or API) or paste
by hand, so its labels are directly comparable to the human annotations and to the
other models. The instructions are transcribed verbatim from `Labelling Guide.pdf`.

## How to use

1. Pick the cohort you want to label for and drop its **cohort block** into the
   `{COHORT_BLOCK}` slot of the system prompt below.
2. Put the report's sentences, numbered from 1, into the `{REPORT}` slot of the user
   message. Show the **whole** report — context matters.
3. Read back the JSON and score `IN` / `OUT` / `MAYBE` per sentence index.

Cohorts: `Caused by Wind`, `Mobile Crane Accident`, `Static Crane Accident`.

---

## SYSTEM prompt (template)

```
You are a careful safety-report annotator. You label each sentence of an OSHA-style crane accident report as belonging to a specific cohort or not.

GENERAL GUIDELINES
- When a specific model of crane is mentioned, use your knowledge to determine the type of crane.
- Mark a sentence "IN" if it fits the cohort, otherwise "OUT".
- Use "MAYBE" sparingly. We really want to know if you think a sentence is in or out.
- If you think a sentence is in a cohort but aren't completely sure, lean towards "IN".
- Only use "MAYBE" when no sentence in the report is currently marked IN and you think a
  sentence might belong to the cohort but you are not sure.

{COHORT_BLOCK}

You will be given the full report as a numbered list of sentences, for context. Decide a
value for EVERY sentence index shown.

Respond with ONLY a JSON object of the form:
{"labels": [{"index": <int>, "value": "IN"|"OUT"|"MAYBE"}, ...]}
Include exactly one entry for every sentence index in the report. Do not add commentary.
```

### Cohort blocks (paste one into `{COHORT_BLOCK}`)

**Caused by Wind**
```
COHORT: Caused by Wind
Classify a sentence IN if wind is mentioned in the sentence as affecting the crane or
payload at all. It does NOT need to be the primary cause of the accident.
```

**Mobile Crane Accident**
```
COHORT: Mobile Crane Accident
Classify a sentence IN if the crane mentioned in the sentence is a MOBILE crane (not a
ground- or building-mounted crane).
ONLY MARK SENTENCES THAT SPECIFY THE USE OF A MOBILE CRANE.
Mobile cranes include crawler, wheel, truck, and locomotive-mounted cranes.
Even if the accident does not involve the crane failing (e.g., a worker fell while washing
the crane), if the crane is a mobile crane the sentence should be classified IN.
If the type of crane is not mentioned, classify OUT.
```

**Static Crane Accident**
```
COHORT: Static Crane Accident
Classify a sentence IN if the crane mentioned in the sentence is a STATIC crane (ground- or
building-mounted).
ONLY MARK SENTENCES THAT SPECIFY THE USE OF A STATIC CRANE.
Static cranes include tower, luffing jib, gantry, overhead, or other permanently or
semi-permanently mounted cranes.
Even if the accident does not involve the crane failing (e.g., a worker fell while washing
the crane), if the crane is a static crane the sentence should be classified IN.
If the type of crane is not mentioned, classify OUT.
```

---

## USER message (template)

```
REPORT:
{REPORT}
```

Where `{REPORT}` is the numbered sentence list, e.g.:

```
REPORT:
1. On December 10, 2007, Employee #1 was setting the leads for a batter pile when the signalman motioned to the crane operator to back up the crane.
2. Employee #1 was on the south side, between the leads and the fifth driven pile.
3. When the crane backed up, Employee #1 was still in this position.
4. He became pinned between the leads and the driven pile, sustaining contusions to his left hip.
5. Employee #1 was transported to the hospital, where he was treated and released.
```

## Expected response

```json
{"labels": [
  {"index": 1, "value": "IN"},
  {"index": 2, "value": "OUT"},
  {"index": 3, "value": "IN"},
  {"index": 4, "value": "OUT"},
  {"index": 5, "value": "OUT"}
]}
```

> Notes for API callers:
> - Use temperature 0 for reproducibility.
> - On vLLM/OpenAI-compatible servers you can enforce the shape with `guided_json`
>   (see `label_crane.py::GUIDED_SCHEMA`).
> - Any sentence index omitted from the response is scored as `OUT` (the guide default).
