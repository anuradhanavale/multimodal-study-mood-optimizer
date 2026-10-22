"""Rule-based, traceable recommendation engine (no LLM dependency)."""
TECH = {
    0: [("Active recall problem sets", "Attempt practice questions without notes, then check answers."),
        ("Feynman explanation", "Explain the hardest concept aloud as if teaching someone."),
        ("Interleaved practice", "Mix 2-3 topic types inside one block.")],
    1: [("Pomodoro with early break", "One short focused sprint, then a real break away from screens."),
        ("Walk-and-review", "Take a 5-minute walk, then skim summary notes."),
        ("Flashcard sprint", "Quick low-effort recall with flashcards.")],
    2: [("Concept skim + one example", "Read one concept and work through one worked example."),
        ("Mind-map existing notes", "Reorganise notes visually - low cognitive load."),
        ("Tidy and plan", "Organise materials and plan your next session.")]}

def reasons(c):
    r = []
    if c["energy"] <= 2: r.append(f"Low self-reported energy ({c['energy']}/5)")
    if c["focus"] <= 2: r.append(f"Low self-reported focus ({c['focus']}/5)")
    if c["difficulty"] >= 4: r.append(f"High task difficulty ({c['difficulty']}/5)")
    if c["sleep_hours"] < 6: r.append(f"Short sleep ({c['sleep_hours']:.1f} h)")
    if c["energy"] >= 4 and c["focus"] >= 4: r.append("High energy and focus")
    return r or ["No single strong signal; based on the combined pattern"]

def recommend(label, proba, checkin, feedback_scores=None):
    minutes, diff = int(checkin["minutes_available"]), checkin["difficulty"]
    work, brk = {0: (50 if diff <= 3 else 40, 10), 1: (20, 10), 2: (15, 5)}[label]
    work = min(work, minutes)
    brk = min(brk, max(0, minutes - work))
    blocks = max(1, min(3, (minutes + brk) // (work + brk))) if work + brk else 1
    total = blocks * work + (blocks - 1) * brk
    if total > minutes:
        blocks, total = 1, work
    scores = feedback_scores or {}
    tech, desc = max(TECH[label], key=lambda t: scores.get(t[0], 3.0))
    return {"technique": tech, "description": desc, "work_minutes": work, "break_minutes": brk, "blocks": blocks,
            "total_minutes": total, "low_confidence": bool(max(proba) < 0.5), "reasons": reasons(checkin)}
