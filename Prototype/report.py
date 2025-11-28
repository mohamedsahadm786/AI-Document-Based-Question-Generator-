#!/usr/bin/env python3
"""
report.py - robust reporter that also reads answer_key_*.json fallback files.

Input on stdin:
  { "action":"report", "test": {...}, "submission": {...} }

Output JSON:
  { "ok": true, "report": "<text>", "debug": {...} }
"""

import sys, json, os, glob, re, traceback

# ------------------------
# Helpers
# ------------------------
def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def find_saved_test_by_id(test_id):
    folder = os.path.join(os.getcwd(), "generated_tests")
    if not os.path.isdir(folder):
        return None
    for p in glob.glob(os.path.join(folder, "test_*.json")):
        data = load_json(p)
        if isinstance(data, dict) and data.get("test_id") == test_id:
            return p, data
    return None

def collect_answer_key_files():
    """
    Read all answer_key_*.json files in generated_tests and merge into a dict.
    Returns merged map and list of file paths read.
    """
    folder = os.path.join(os.getcwd(), "generated_tests")
    merged = {}
    files_read = []
    if not os.path.isdir(folder):
        return merged, files_read
    for p in glob.glob(os.path.join(folder, "answer_key_*.json")):
        data = load_json(p)
        if isinstance(data, dict):
            merged.update(data)
            files_read.append(p)
    return merged, files_read

def resolve_model_answer(test, q, merged_answer_key):
    # priority: q['answer'] -> test['answer_key'][qid] -> merged_answer_key[qid]
    if q.get("answer") is not None:
        return q.get("answer")
    for k in ("correct","correct_answer","correct_opt","correct_option"):
        if q.get(k) is not None:
            return q.get(k)
    ak = (test.get("answer_key") or {})
    qid = q.get("id") or q.get("qid")
    if qid and qid in ak:
        return ak[qid]
    if qid and qid in merged_answer_key:
        return merged_answer_key[qid]
    # last resort: q.get('reference_answer')
    if q.get("reference_answer") is not None:
        return q.get("reference_answer")
    return None

def get_stem(q):
    return q.get("stem") or q.get("question") or q.get("prompt") or q.get("title") or ""

def simple_overlap_score(ref, stu):
    if not (ref or "").strip():
        return None, "(no reference available)"
    if not (stu or "").strip():
        return 0, "No answer submitted."
    ref_tokens = set(re.sub(r"[^a-z0-9\s]", " ", (ref or "").lower()).split())
    stu_tokens = set(re.sub(r"[^a-z0-9\s]", " ", (stu or "").lower()).split())
    if not ref_tokens:
        return None, "(no reference tokens)"
    common = ref_tokens.intersection(stu_tokens)
    score = int(round(len(common) / max(1, len(ref_tokens)) * 100))
    fb = "Good." if score >= 85 else ("Partial." if score >= 60 else ("Weak." if score > 0 else "Irrelevant."))
    return score, fb

# ------------------------
# Main generation
# ------------------------
def build_report_and_debug(test, submission):
    debug = {
        "received_test_id": test.get("test_id"),
        "incoming_has_answer_key": bool(test.get("answer_key")),
        "incoming_questions_have_answers": any(q.get("answer") for q in test.get("questions", [])),
        "merged_from_test_file": False,
        "merged_from_answer_key_files": False,
        "merged_answer_key_files": [],
        "resolved_answers": {}
    }

    # merged_answer_key will contain mappings from any answer_key_*.json files
    merged_answer_key, ak_files = collect_answer_key_files()
    if ak_files:
        debug["merged_from_answer_key_files"] = True
        debug["merged_answer_key_files"] = ak_files

    # If the incoming test lacks answers, attempt to find saved test and merge
    if not test.get("answer_key") or not debug["incoming_questions_have_answers"]:
        tid = test.get("test_id")
        if tid:
            found = find_saved_test_by_id(tid)
            if found:
                path, saved = found
                # merge saved answer_key into incoming test if missing
                saved_ak = saved.get("answer_key") or {}
                if saved_ak:
                    ak = test.setdefault("answer_key", {})
                    for k,v in saved_ak.items():
                        if k not in ak:
                            ak[k] = v
                # merge per-question 'answer'
                saved_qs = {q.get("id"): q for q in saved.get("questions", []) if q.get("id")}
                for q in test.get("questions", []):
                    sid = q.get("id")
                    if sid and q.get("answer") is None and sid in saved_qs:
                        if saved_qs[sid].get("answer") is not None:
                            q["answer"] = saved_qs[sid].get("answer")
                debug["merged_from_test_file"] = True

    # Build textual report and resolved answer debug map
    lines = []
    lines.append("DETAILED PER-QUESTION EVALUATION")
    lines.append("=" * 40)

    for idx, q in enumerate(test.get("questions", []), start=1):
        try:
            qid = q.get("id") or q.get("qid") or f"q_{idx}"
            qtype = q.get("type", "unknown")
            stem = get_stem(q).strip() or "(no question text)"
            student_ans = ""
            if isinstance(submission, dict):
                student_ans = submission.get(qid, "")

            model_ans = resolve_model_answer(test, q, merged_answer_key)
            debug["resolved_answers"][qid] = model_ans

            lines.append("")
            lines.append(f"Question {idx}: {stem}")
            lines.append("-" * 40)

            if qtype == "mcq":
                opts = q.get("options") or []
                if opts:
                    for i,opt in enumerate(opts):
                        label = chr(ord("A") + i) if i < 26 else str(i+1)
                        lines.append(f"  {label}) {opt}")
                lines.append(f"Model answer: {model_ans if model_ans is not None else '(not available)'}")
                lines.append(f"Your reply: {student_ans if student_ans else ''}")
                correct = False
                if model_ans is not None:
                    try:
                        correct = str(student_ans).strip().lower() == str(model_ans).strip().lower()
                    except:
                        correct = False
                lines.append(f"Is correct: {'Yes' if correct else 'No'}")
                lines.append(f"Marks: {1 if correct else 0}")

            elif qtype == "yesno":
                lines.append(f"Model answer: {model_ans if model_ans is not None else '(not available)'}")
                lines.append(f"Your reply: {student_ans if student_ans else ''}")
                correct = False
                if model_ans is not None:
                    try:
                        correct = str(student_ans).strip().lower() == str(model_ans).strip().lower()
                    except:
                        correct = False
                lines.append(f"Is correct: {'Yes' if correct else 'No'}")
                lines.append(f"Marks: {1 if correct else 0}")

            elif qtype == "descriptive":
                ref = q.get("reference_answer") or model_ans or ""
                lines.append("Model answer:")
                if ref:
                    for ln in str(ref).splitlines():
                        lines.append("  " + ln)
                else:
                    lines.append("  (no reference available)")
                lines.append("Your reply:")
                if student_ans:
                    for ln in str(student_ans).splitlines():
                        lines.append("  " + ln)
                else:
                    lines.append("  (no answer submitted)")
                score, fb = simple_overlap_score(ref, student_ans)
                lines.append("Analysis:")
                lines.append("  " + (fb or "(no feedback)"))
                lines.append(f"Score: {score if score is not None else 'N/A'}")
                lines.append(f"Pass: {'Yes' if (score is not None and score >= 60) else 'No'}")

            elif qtype == "code":
                lines.append("Note: Code question — will evaluate later and get back to you.")
                lines.append("Your submission:")
                if student_ans:
                    for ln in str(student_ans).splitlines():
                        lines.append("  " + ln)
                else:
                    lines.append("  (no code submitted)")

            else:
                lines.append(f"Question type: {qtype}")
                lines.append(f"Model answer: {model_ans if model_ans is not None else '(not available)'}")
                lines.append(f"Your reply: {student_ans if student_ans else ''}")

        except Exception as ex:
            lines.append(f"\nQuestion {idx}: (failed to render — error: {ex})")
            lines.append("  (skipping)")

    return "\n".join(lines), debug

# ------------------------
# Entrypoint
# ------------------------
def main():
    try:
        raw = sys.stdin.read()
        if not raw:
            print(json.dumps({"ok": False, "error": "No input on stdin"}))
            return
        payload = json.loads(raw)
        if payload.get("action") != "report":
            print(json.dumps({"ok": False, "error": "Unsupported action"}))
            return
        test = payload.get("test", {}) or {}
        submission = payload.get("submission", {}) or {}
        report_text, debug = build_report_and_debug(test, submission)
        out = {"ok": True, "report": report_text, "debug": debug}
        print(json.dumps(out))
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e), "trace": traceback.format_exc()}))

if __name__ == "__main__":
    main()
