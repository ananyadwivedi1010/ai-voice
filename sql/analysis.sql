-- analysis.sql — Analytical queries for comparing prompt versions.
-- These queries are loaded by report.py for the comparison table.

-- ============================================================================
-- QUERY: outcome_by_version
-- Outcome distribution (callback_booked / declined / dropped) by version
-- ============================================================================
SELECT
    prompt_version,
    outcome,
    COUNT(*) as count
FROM conversations
GROUP BY prompt_version, outcome
ORDER BY prompt_version, outcome;


-- ============================================================================
-- QUERY: compliance_by_version
-- Compliance failure rate by prompt version and persona
-- Shows % of conversations with forbidden phrases (guaranteed, rates, etc.)
-- ============================================================================
SELECT
    c.prompt_version,
    c.persona,
    ROUND(AVG(s.compliance_fail) * 100, 1) as compliance_fail_pct
FROM conversations c
JOIN scores s ON c.id = s.conversation_id
GROUP BY c.prompt_version, c.persona
ORDER BY c.prompt_version, c.persona;


-- ============================================================================
-- QUERY: words_and_turns
-- Average reply word count and average turns per conversation by version
-- Target: avg_reply_words should be under 35 for voice-first design
-- ============================================================================
SELECT
    c.prompt_version,
    ROUND(AVG(s.avg_reply_words), 1) as avg_words_per_reply,
    ROUND(AVG(c.num_turns), 1) as avg_turns,
    MIN(s.avg_reply_words) as min_words,
    MAX(s.avg_reply_words) as max_words
FROM conversations c
JOIN scores s ON c.id = s.conversation_id
GROUP BY c.prompt_version
ORDER BY c.prompt_version;


-- ============================================================================
-- QUERY: drop_off_analysis
-- Drop-off analysis: which turn_idx do conversations end at?
-- Groups by persona to see where each persona typically stops engaging.
-- ============================================================================
SELECT
    c.persona,
    t.turn_idx,
    COUNT(*) as conversations_ending_at_turn
FROM conversations c
JOIN turns t ON c.id = t.conversation_id
WHERE t.turn_idx = (
    SELECT MAX(turn_idx)
    FROM turns
    WHERE conversation_id = c.id
)
GROUP BY c.persona, t.turn_idx
ORDER BY c.persona, t.turn_idx;


-- ============================================================================
-- QUERY: qualification_rate
-- Qualification completion rate by persona for the latest version
-- Shows which personas are harder to qualify (e.g., busy, skeptical)
-- ============================================================================
SELECT
    c.persona,
    COUNT(*) as total_conversations,
    SUM(s.qualification_complete) as qualified_count,
    ROUND(AVG(s.qualification_complete) * 100, 1) as qualification_pct
FROM conversations c
JOIN scores s ON c.id = s.conversation_id
WHERE c.prompt_version = (
    SELECT prompt_version
    FROM conversations
    ORDER BY created_at DESC
    LIMIT 1
)
GROUP BY c.persona
ORDER BY qualification_pct DESC;


-- ============================================================================
-- ADDITIONAL: Language match by version (LLM judge metric)
-- ============================================================================
SELECT
    c.prompt_version,
    COUNT(*) as total_scored,
    SUM(s.language_match) as matched_count,
    ROUND(AVG(s.language_match) * 100, 1) as language_match_pct
FROM conversations c
JOIN scores s ON c.id = s.conversation_id
WHERE s.language_match IS NOT NULL
GROUP BY c.prompt_version
ORDER BY c.prompt_version;


-- ============================================================================
-- ADDITIONAL: Objection handling success rate by version
-- ============================================================================
SELECT
    c.prompt_version,
    s.objection_handled,
    COUNT(*) as count
FROM conversations c
JOIN scores s ON c.id = s.conversation_id
WHERE s.objection_handled IS NOT NULL
GROUP BY c.prompt_version, s.objection_handled
ORDER BY c.prompt_version, s.objection_handled;
