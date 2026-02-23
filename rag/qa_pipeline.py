# =============================================================================
# qa_pipeline.py — STEPS 1, 2 & 4 CHANGES
# =============================================================================
#
# HOW TO USE THIS FILE
# ─────────────────────
# This file contains 4 method replacements and 1 new method.
# Each section is clearly marked with:
#   [ACTION]   — what to do (replace / add new)
#   [LOCATION] — where in qa_pipeline.py to find the target
#   [PASTE]    — the exact code to paste
#
# ORDER OF CHANGES:
#   CHANGE A — Replace _classify_question_type             (Step 1)
#   CHANGE B — Add new method _keyword_preclassify         (Step 1)
#   CHANGE C — Add new method _reclassify_if_evidence_question  (Step 1)
#   CHANGE D — Replace _create_type_specific_prompt        (Step 2 + Step 4)
#
# =============================================================================


# =============================================================================
# CHANGE A
# [ACTION]   Replace the entire _classify_question_type method
# [LOCATION] Search for "def _classify_question_type" in qa_pipeline.py
#            Delete from "def _classify_question_type" down to its final
#            "return category" line, and paste this in its place.
# =============================================================================

    def _classify_question_type(self, question: str) -> str:
        """
        Classify the question type to use the appropriate answer template.

        Three-layer defence against misclassification:
          Layer 1 — keyword pre-check  (before LLM, deterministic)
          Layer 2 — LLM classifier     (handles ambiguous cases)
          Layer 3 — post-LLM safety nets (reclassify if LLM still wrong)
        """

        # ── LAYER 1: keyword pre-check ────────────────────────────────────────
        # Runs BEFORE the LLM. Returns immediately for obvious patterns.
        # Prevents LLM from misreading "What is the evidence for..." as definition.
        pre_check = self._keyword_preclassify(question)
        if pre_check:
            logger.info(f"Question pre-classified by keyword as: {pre_check}")
            return pre_check
        # ─────────────────────────────────────────────────────────────────────

        # ── LAYER 2: LLM classifier ───────────────────────────────────────────
        classification_prompt = f"""Classify this thyroid cancer question into ONE category.

Categories:
- definition:    "What is X?", "Tell me about X", "Explain X", "Describe X"
- complications: "What are complications/risks/side effects of X?", "What risks?"
- comparison:    "X vs Y", "difference between X and Y", "compare X and Y"
- treatment:     "How to treat X?", "What are treatment options?", "How is X treated?"
- diagnosis:     "How is X diagnosed?", "What tests for X?", "How to detect X?"
- timing:        "When should X?", "When is X recommended?", "When to do X?"
- evidence:      "What is the evidence for X?", "What does research/trials show?",
                 "What data exists for X?", "Is X effective?", "Efficacy of X?",
                 "What are outcomes of X?", "What studies support X?"

Question: {question}

IMPORTANT: If the question contains "evidence", "efficacy", "trial", "outcomes",
"research shows", or "studies show" — classify as "evidence", not "definition".

Return ONLY the category name (one word), nothing else:"""

        try:
            category = self.llm.ask(classification_prompt).strip().lower()

            valid_categories = [
                "definition", "complications", "comparison",
                "treatment", "diagnosis", "timing", "evidence"
            ]
            if category not in valid_categories:
                logger.warning(f"Invalid category '{category}', defaulting to 'definition'")
                category = "definition"

            # ── LAYER 3: post-LLM safety nets ─────────────────────────────────
            # Only run if LLM returned definition — it's the most commonly wrong label
            if category == "definition":
                category = self._reclassify_if_diagnostic_tool(question, category)
            if category == "definition":
                category = self._reclassify_if_evidence_question(question, category)
            # ─────────────────────────────────────────────────────────────────

            logger.info(f"Question classified as: {category}")
            return category

        except Exception as e:
            logger.error(f"Error classifying question: {e}")
            return "definition"


# =============================================================================
# CHANGE B
# [ACTION]   Add this as a NEW method
# [LOCATION] Paste directly AFTER _classify_question_type and BEFORE
#            _reclassify_if_diagnostic_tool in qa_pipeline.py
# =============================================================================

    def _keyword_preclassify(self, question: str) -> Optional[str]:
        """
        Fast keyword-based pre-classification that bypasses the LLM entirely.

        Returns a category string if confident, or None to fall through to LLM.
        Catches patterns the LLM consistently gets wrong — particularly evidence
        questions that start with "What is..." being labelled as definition.
        """
        q = question.lower().strip()

        # ── Evidence ──────────────────────────────────────────────────────────
        # These phrases unambiguously signal an evidence/trial synthesis question
        evidence_trigger_phrases = [
            "what is the evidence",
            "what is the clinical evidence",
            "what evidence exists",
            "what does the evidence",
            "what does research show",
            "what does the research show",
            "what do studies show",
            "what do the studies show",
            "what clinical trials",
            "what trials exist",
            "what trials support",
            "evidence for",
            "evidence of",
            "evidence supporting",
            "evidence behind",
            "evidence base for",
            "evidence-based",
            "is there evidence",
            "what are the outcomes of",
            "what are clinical outcomes",
            "efficacy of",
            "how effective is",
            "how efficacious",
            "what is the efficacy",
            "clinical evidence for",
            "trial data",
            "trial evidence",
        ]
        if any(phrase in q for phrase in evidence_trigger_phrases):
            return "evidence"

        # ── Complications ─────────────────────────────────────────────────────
        complication_trigger_phrases = [
            "what are the complications",
            "what are complications",
            "what are the risks of",
            "what are risks of",
            "what are the side effects",
            "what are side effects",
            "what are adverse",
            "side effects of",
            "risks of",
            "complications of",
            "adverse effects of",
            "adverse events of",
        ]
        if any(phrase in q for phrase in complication_trigger_phrases):
            return "complications"

        # ── Comparison ────────────────────────────────────────────────────────
        comparison_trigger_phrases = [" vs ", " versus ", "difference between", "compare "]
        if any(phrase in q for phrase in comparison_trigger_phrases):
            return "comparison"

        # ── Timing ────────────────────────────────────────────────────────────
        timing_trigger_phrases = [
            "when should", "when is", "when to ", "when do ",
            "when are", "when would",
        ]
        if any(q.startswith(phrase) or f" {phrase}" in q for phrase in timing_trigger_phrases):
            return "timing"

        # Not confident — let LLM decide
        return None


# =============================================================================
# CHANGE C
# [ACTION]   Add this as a NEW method
# [LOCATION] Paste directly AFTER _reclassify_if_diagnostic_tool in qa_pipeline.py
#            (that method already exists — add this one right below it)
# =============================================================================

    def _reclassify_if_evidence_question(self, question: str, original_category: str) -> str:
        """
        Post-LLM safety net for evidence questions.

        Mirrors the existing _reclassify_if_diagnostic_tool pattern.
        If the LLM returned 'definition' but the question is clearly about
        evidence, trials, or efficacy — override to 'evidence'.
        """
        q = question.lower()

        # Strong phrases → always override
        strong_evidence_phrases = [
            "what is the evidence",
            "evidence for",
            "evidence of",
            "evidence on",
            "is there evidence",
            "what trials",
            "efficacy of",
            "how effective is",
            "what studies",
            "what research",
            "clinical evidence",
            "trial data",
        ]
        if any(phrase in q for phrase in strong_evidence_phrases):
            logger.info(
                f"Reclassifying '{question}' from 'definition' → 'evidence' "
                f"(strong evidence phrase matched in post-LLM check)"
            )
            return "evidence"

        # Weak indicators — only override if 2+ present together
        weak_evidence_indicators = [
            "evidence", "trial", "study", "studies", "research",
            "efficacy", "effective", "outcome", "outcomes",
            "survival", "response rate", "pfs", "os",
            "randomized", "systematic review", "meta-analysis",
            "phase 3", "phase 2", "rct",
        ]
        matches = sum(1 for indicator in weak_evidence_indicators if indicator in q)
        if matches >= 2:
            logger.info(
                f"Reclassifying '{question}' from 'definition' → 'evidence' "
                f"({matches} weak evidence indicators matched)"
            )
            return "evidence"

        return original_category


# =============================================================================
# CHANGE D
# [ACTION]   Replace the entire _create_type_specific_prompt method
# [LOCATION] Search for "def _create_type_specific_prompt" in qa_pipeline.py
#            Delete from that line down to the final "return" of the method,
#            and paste this in its place.
#
# What changed vs the original:
#   Step 2 — Added "evidence" template (cancer-subtype structure, trial names,
#             numerical outcomes, Key Considerations block)
#   Step 4 — Added STRICT SCHEMA RULES block to every template:
#             • null for missing data — never invent content
#             • required fields labelled explicitly
#             • numerical outcomes required in evidence template
#             • placeholder text like "[Topic]" is forbidden
# =============================================================================

    def _create_type_specific_prompt(
        self,
        question: str,
        context: str,
        question_type: str
    ) -> str:
        """
        Build a question-type-specific prompt that returns structured JSON.
        Every template includes a strict schema contract (Step 4) to prevent
        invented content, placeholder text, and missing required fields.
        """

        base_instructions = f"""
{self.instructions}

You are a medical information assistant specialised in thyroid cancer.
Answer using ONLY the tagged excerpts below. Do NOT use any outside knowledge.

SOURCE TAGGING RULES:
- Every factual claim must end with its source tag: "Example fact [SOURCE_1]."
- Combine tags when merging facts from multiple sources: "Fact [SOURCE_1][SOURCE_3]."
- Never write a factual sentence without at least one [SOURCE_X] tag.

STRICT SCHEMA RULES — READ BEFORE WRITING:
1. NEVER invent content. If a field has no supporting data in the context, set it to null.
2. NEVER use placeholder text like "[Topic]", "[Drug Name]", "[X]", "[Insert here]".
3. NEVER copy section names verbatim from the template if they don't apply — use null.
4. ALL string values must be real sentences, not template placeholders.
5. Return ONLY valid JSON. No markdown fences, no explanation, no extra text.

CONTEXT WITH SOURCE TAGS:
{context}

"""

        # ── EVIDENCE ─────────────────────────────────────────────────────────
        # Step 2: New template — structured by cancer subtype, forces trial names
        # and numerical outcomes, ends with Key Considerations block.
        if question_type == "evidence":
            return base_instructions + f"""
QUESTION: {question}

You are synthesising clinical trial evidence. Your job is NOT to summarise sources —
it is to construct a unified expert answer organised by cancer subtype.

EVIDENCE TEMPLATE RULES (Step 4 additions):
- Every drug item MUST include the trial name if one appears in the context
  (e.g. SELECT trial, DECISION trial, ZETA trial).
- Every drug item MUST include at least one numerical outcome if one appears
  in the context (e.g. median PFS, HR, ORR, OS in months).
- If no trial name or number exists in the context for a drug, set that drug's
  "description" to null — do not invent statistics.
- Omit entire sections (set to null) if the context contains no relevant data
  for that cancer type.
- The "Key Considerations" section is REQUIRED and must have at least 2 items.

Return this exact JSON structure:

{{
  "overview": "2-3 sentences covering: what TKIs are, their general role in advanced thyroid cancer, approval status, and overall evidence quality. [SOURCE_X] tags required. STRING — not null.",

  "sections": [

    {{
      "header": "Evidence in Differentiated Thyroid Cancer (DTC)",
      "items": [
        {{
          "name": "Drug name exactly as written in context (e.g. Lenvatinib). REQUIRED. null if unknown.",
          "description": "Trial name + primary endpoint with exact numbers from context (e.g. SELECT trial: median PFS 18.3 vs 3.6 months, HR 0.21, p<0.001). Approval status. [SOURCE_X]. null if no numerical data in context.",
          "highlight": "One sentence on what makes this agent clinically significant or preferred. [SOURCE_X]. null if not in context."
        }}
      ]
    }},

    {{
      "header": "Evidence in Medullary Thyroid Cancer (MTC)",
      "items": [
        {{
          "name": "Drug name. REQUIRED. null if not in context.",
          "description": "Trial name, primary endpoint numbers, mutation targets (e.g. RET), approval status. [SOURCE_X]. null if no data.",
          "highlight": "One sentence on clinical positioning or mutation relevance. [SOURCE_X]. null if not in context."
        }}
      ]
    }},

    {{
      "header": "Evidence in Anaplastic Thyroid Cancer (ATC)",
      "items": [
        {{
          "name": "Drug or combination name (e.g. Dabrafenib + Trametinib). null if not in context.",
          "description": "Evidence strength, key outcomes (OS, response rate) with numbers if available, why TKI monotherapy is limited. [SOURCE_X]. null if no data.",
          "highlight": "Key limitation or reason combination therapy is preferred. [SOURCE_X]. null if not in context."
        }}
      ]
    }},

    {{
      "header": "Key Considerations",
      "items": [
        {{
          "consideration": "Toxicity Profile",
          "description": "Rate of grade ≥3 adverse events, most common AEs (e.g. hypertension, hand-foot syndrome, fatigue) with percentages if in context. [SOURCE_X]. null if not in context."
        }},
        {{
          "consideration": "Resistance",
          "description": "When and why resistance develops, salvage options if mentioned. [SOURCE_X]. null if not in context."
        }},
        {{
          "consideration": "Patient Selection",
          "description": "Criteria for initiating TKI therapy: radioiodine-refractory, symptomatic, rapidly progressive disease. [SOURCE_X]. null if not in context."
        }},
        {{
          "consideration": "Treatment Sequencing",
          "description": "First-line vs second-line positioning, role of watchful waiting. [SOURCE_X]. null if not in context."
        }}
      ]
    }}

  ]
}}

Return ONLY valid JSON, no other text:"""

        # ── DEFINITION ────────────────────────────────────────────────────────
        elif question_type == "definition":
            return base_instructions + f"""
QUESTION: {question}

DEFINITION TEMPLATE RULES (Step 4):
- "Types" section: use the actual type names from context. NEVER write "Type Name"
  or "[Topic]" as placeholders. If types are not in context, set items to null.
- All section content must come from context. Null if not present.

Return this exact JSON structure:

{{
  "overview": "2-3 sentence definition: what it is, who it affects, general outlook. [SOURCE_X]. REQUIRED — not null.",

  "sections": [
    {{
      "header": "Types",
      "items": [
        {{
          "name": "Actual type name from context (e.g. Papillary Thyroid Cancer). null if not in context.",
          "description": "What distinguishes this type, incidence if mentioned. [SOURCE_X]. null if not in context.",
          "details": "Prognosis, survival rates, or key statistics if in context. [SOURCE_X]. null if not in context."
        }}
      ]
    }},
    {{
      "header": "Causes and Risk Factors",
      "content": "Paragraph on known causes and risk factors from context. [SOURCE_X]. null if not in context."
    }},
    {{
      "header": "Common Symptoms",
      "items": [
        {{
          "symptom": "Actual symptom name from context. null if not in context.",
          "description": "Explanation of why it occurs or how it presents. [SOURCE_X]. null if not in context."
        }}
      ]
    }},
    {{
      "header": "Diagnosis and Treatment",
      "content": "Paragraph on how it is diagnosed and treated based on context. [SOURCE_X]. null if not in context."
    }}
  ]
}}

Return ONLY valid JSON, no other text:"""

        # ── COMPLICATIONS ─────────────────────────────────────────────────────
        elif question_type == "complications":
            return base_instructions + f"""
QUESTION: {question}

COMPLICATIONS TEMPLATE RULES (Step 4):
- Use real complication names from context, never generic placeholders.
- Include frequency/percentage data if present in context.
- Set any section with no supporting context data to null entirely.

Return this exact JSON structure:

{{
  "overview": "2-3 sentence summary of the complication landscape for this procedure/treatment. [SOURCE_X]. REQUIRED.",

  "sections": [
    {{
      "header": "Common and Temporary Complications",
      "items": [
        {{
          "complication": "Real complication name from context. null if not present.",
          "description": "Explanation, mechanism, and frequency if mentioned. [SOURCE_X]. null if not in context.",
          "frequency": "Percentage or rate from context (e.g. 30-40% of cases). null if not stated."
        }}
      ]
    }},
    {{
      "header": "Less Common or Potentially Permanent Complications",
      "items": [
        {{
          "complication": "Real complication name from context. null if not present.",
          "description": "Explanation and any long-term impact. [SOURCE_X]. null if not in context."
        }}
      ]
    }},
    {{
      "header": "Rare but Serious Complications",
      "items": [
        {{
          "complication": "Real complication name from context. null if not present.",
          "description": "Why it is serious and how it is managed if mentioned. [SOURCE_X]. null if not in context."
        }}
      ]
    }},
    {{
      "header": "Management and Prevention",
      "content": "Paragraph on how complications are managed or prevented based on context. [SOURCE_X]. null if not in context."
    }}
  ]
}}

Return ONLY valid JSON, no other text:"""

        # ── COMPARISON ────────────────────────────────────────────────────────
        elif question_type == "comparison":
            return base_instructions + f"""
QUESTION: {question}

COMPARISON TEMPLATE RULES (Step 4):
- Replace option_a_label and option_b_label with the ACTUAL names of the two things
  being compared (extract from the question itself).
- Only include aspects where context has data for BOTH options. Null otherwise.

Return this exact JSON structure:

{{
  "overview": "2-3 sentence comparison summary explaining what is being compared and the clinical context. [SOURCE_X]. REQUIRED.",

  "sections": [
    {{
      "header": "Key Differences",
      "option_a_label": "Name of first option exactly as in the question.",
      "option_b_label": "Name of second option exactly as in the question.",
      "comparison_table": [
        {{
          "aspect": "Specific aspect being compared (e.g. Mechanism of Action, PFS, Toxicity). null if not in context.",
          "option_a": "What the first option does for this aspect, with numbers if available. [SOURCE_X]. null if not in context.",
          "option_b": "What the second option does for this aspect, with numbers if available. [SOURCE_X]. null if not in context."
        }}
      ]
    }},
    {{
      "header": "Shared Characteristics",
      "content": "Paragraph on what both options have in common. [SOURCE_X]. null if not in context."
    }},
    {{
      "header": "Clinical Outcomes",
      "content": "Paragraph comparing survival, response rates, or other outcomes with numbers if available. [SOURCE_X]. null if not in context."
    }}
  ]
}}

Return ONLY valid JSON, no other text:"""

        # ── TREATMENT ─────────────────────────────────────────────────────────
        elif question_type == "treatment":
            return base_instructions + f"""
QUESTION: {question}

TREATMENT TEMPLATE RULES (Step 4):
- Use real treatment names from context. Never write "Treatment Name" as a placeholder.
- Include evidence grade (RCT, guideline, observational) if discernible from context.
- Set sections to null if no supporting context data exists for them.

Return this exact JSON structure:

{{
  "overview": "2-3 sentence treatment landscape summary with guideline-recommended first-line approaches. [SOURCE_X]. REQUIRED.",

  "sections": [
    {{
      "header": "First-Line Treatments",
      "items": [
        {{
          "treatment": "Real treatment or drug name from context. null if not present.",
          "description": "Indication, mechanism, key trial result with numbers if available, approval status. [SOURCE_X]. null if not in context.",
          "evidence_grade": "e.g. Phase 3 RCT / Guideline recommendation / Observational. null if not discernible."
        }}
      ]
    }},
    {{
      "header": "Second-Line and Salvage Treatments",
      "items": [
        {{
          "treatment": "Real treatment name from context. null if not present.",
          "description": "When used, outcomes, any key trial with numbers. [SOURCE_X]. null if not in context."
        }}
      ]
    }},
    {{
      "header": "Treatment by Cancer Subtype",
      "content": "Paragraph mapping approved agents to DTC, MTC, ATC, and PDTC subtypes where context supports it. [SOURCE_X]. null if not in context."
    }},
    {{
      "header": "Key Considerations",
      "items": [
        {{
          "consideration": "Real consideration title from context (e.g. Monitoring, Resistance). null if not present.",
          "description": "Concise explanation with any data. [SOURCE_X]. null if not in context."
        }}
      ]
    }}
  ]
}}

Return ONLY valid JSON, no other text:"""

        # ── DIAGNOSIS ─────────────────────────────────────────────────────────
        elif question_type == "diagnosis":
            return base_instructions + f"""
QUESTION: {question}

DIAGNOSIS TEMPLATE RULES (Step 4):
- Use real procedure names from context. Never write "Procedure Name" as a placeholder.
- Include accuracy/sensitivity/specificity data if present in context.
- Set sections to null if no supporting context data.

Return this exact JSON structure:

{{
  "overview": "2-3 sentence summary of the diagnostic approach and why it is used. [SOURCE_X]. REQUIRED.",

  "sections": [
    {{
      "header": "Key Diagnostic Procedures",
      "items": [
        {{
          "procedure": "Real procedure name from context (e.g. Ultrasound, FNAB). null if not present.",
          "description": "What it does, when it is used, clinical role. [SOURCE_X]. null if not in context.",
          "accuracy": "Sensitivity, specificity, or accuracy percentage if stated in context. null if not stated."
        }}
      ]
    }},
    {{
      "header": "Diagnostic Pathway",
      "content": "Paragraph describing the step-by-step diagnostic process from initial presentation through to diagnosis. [SOURCE_X]. null if not in context."
    }},
    {{
      "header": "Limitations and Considerations",
      "content": "Paragraph on known limitations of the diagnostic approach or tools. [SOURCE_X]. null if not in context."
    }}
  ]
}}

Return ONLY valid JSON, no other text:"""

        # ── TIMING ────────────────────────────────────────────────────────────
        elif question_type == "timing":
            return base_instructions + f"""
QUESTION: {question}

TIMING TEMPLATE RULES (Step 4):
- Use real clinical indications from context. Never write "Situation" as a placeholder.
- Set sections to null if no supporting context data.

Return this exact JSON structure:

{{
  "overview": "2-3 sentence summary of when and why this is recommended, and what guides the decision. [SOURCE_X]. REQUIRED.",

  "sections": [
    {{
      "header": "Key Indications",
      "items": [
        {{
          "indication": "Real clinical situation or patient profile from context. null if not present.",
          "explanation": "Why this timing is recommended in this case, with any supporting data. [SOURCE_X]. null if not in context."
        }}
      ]
    }},
    {{
      "header": "Important Considerations",
      "items": [
        {{
          "consideration": "Real factor affecting timing decisions from context (e.g. tumour size, patient age). null if not present.",
          "description": "How this factor influences the timing recommendation. [SOURCE_X]. null if not in context."
        }}
      ]
    }},
    {{
      "header": "Contraindications or When to Delay",
      "content": "Paragraph on situations where this should be delayed or avoided based on context. [SOURCE_X]. null if not in context."
    }}
  ]
}}

Return ONLY valid JSON, no other text:"""

        # ── FALLBACK ──────────────────────────────────────────────────────────
        else:
            logger.warning(f"Unknown question_type '{question_type}', falling back to definition")
            return self._create_type_specific_prompt(question, context, "definition")
