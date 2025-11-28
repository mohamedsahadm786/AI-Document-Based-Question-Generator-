#!/usr/bin/env python3
# generator.py
# Converted from notebook prototype: programmatic wrapper that accepts JSON on stdin
# and returns JSON on stdout. Uses the original notebook functions with minimal
# edits so you can drop your notebook logic in place without rewriting.

import sys
import os
import json
import uuid
import re
import math
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple

# NOTE: these libraries are heavy. Ensure your environment has them installed.
import fitz  # PyMuPDF
import docx2txt
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from openai import OpenAI
from tqdm import tqdm

# --------- Configuration (edit locally or set env) -------------
# OpenAI key: prefer environment variable; fallback placeholder will raise.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-REPLACE_WITH_YOUR_KEY")
if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("sk-REPLACE"):
    raise RuntimeError("Set OPENAI_API_KEY environment variable (export OPENAI_API_KEY=sk-...)")

client = OpenAI(api_key=OPENAI_API_KEY)

OUTPUT_DIR = Path("./generated_tests")
OUTPUT_DIR.mkdir(exist_ok=True)

CHUNK_CHARS = 3200
CHUNK_OVERLAP = 300
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K_DEFAULT = 4
DEDUP_SIM_THRESHOLD = 0.82
MODEL = "gpt-4o"
TEMPERATURE = 0.0
MAX_TOKENS = 900
DIFF_RATIOS = (0.5, 0.3, 0.2)

# Globals for FAISS & chunk meta
FAISS_INDEX = None
CHUNK_VECTORS = None
CHUNK_META: List[Dict[str, Any]] = []

# ---------------- Utilities ----------------
def now_ts():
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# ---------------- Extraction ----------------
def extract_text_from_pdf(path: str) -> str:
    doc = fitz.open(path)
    pages_text = []
    for i in range(len(doc)):
        page = doc.load_page(i)
        text = page.get_text("text").strip()
        pages_text.append(f"===PAGE {i+1}===\n{text}\n")
    return "\n".join(pages_text)

def extract_text_from_docx(path: str) -> str:
    return docx2txt.process(path)

def extract_text_from_file(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    elif ext in (".docx", ".doc"):
        return extract_text_from_docx(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

# ---------------- Fast paragraph chunker ----------------
def _normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def chunk_text_fast(text: str, chunk_chars: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    if not paragraphs:
        s = _normalize_whitespace(text)
        return [s] if s else []
    chunks: List[str] = []
    cur_parts: List[str] = []
    cur_len = 0
    for p in paragraphs:
        p_len = len(p)
        if p_len > chunk_chars:
            if cur_parts:
                chunks.append(" ".join(cur_parts).strip())
                cur_parts = []
                cur_len = 0
            i = 0
            while i < p_len:
                part = p[i:i+chunk_chars].strip()
                if part:
                    chunks.append(part)
                i += (chunk_chars - overlap) if (chunk_chars - overlap) > 0 else chunk_chars
            continue
        if cur_len + p_len + (1 if cur_parts else 0) <= chunk_chars:
            cur_parts.append(p)
            cur_len += p_len + (1 if cur_parts else 0)
        else:
            if cur_parts:
                chunks.append(" ".join(cur_parts).strip())
            cur_parts = [p]
            cur_len = p_len
    if cur_parts:
        chunks.append(" ".join(cur_parts).strip())
    if overlap > 0 and len(chunks) > 1:
        new_chunks = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = new_chunks[-1]
            take = prev[-overlap:] if len(prev) > overlap else prev
            merged = (take + " " + chunks[i]).strip()
            new_chunks.append(merged)
        chunks = new_chunks
    chunks = [c for c in chunks if len(c) > 50]
    return chunks

# ---------------- TF-IDF keywords per chunk ----------------
def compute_chunk_keywords(chunks_texts: List[str], top_k: int = 6) -> List[List[str]]:
    vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words="english", max_features=5000)
    X = vectorizer.fit_transform(chunks_texts)
    feature_names = np.array(vectorizer.get_feature_names_out())
    keywords_per_chunk = []
    for i in range(X.shape[0]):
        row = X[i].toarray().ravel()
        if row.sum() == 0:
            keywords_per_chunk.append([])
            continue
        top_idx = np.argsort(-row)[:top_k]
        keywords = [feature_names[j] for j in top_idx if row[j] > 0]
        keywords_per_chunk.append(keywords)
    return keywords_per_chunk

# ---------------- Embedding + FAISS ----------------
def build_embeddings_index(chunks: List[Dict[str, Any]], model_name: str = EMBED_MODEL_NAME):
    global FAISS_INDEX, CHUNK_VECTORS
    texts = [c["text"] for c in chunks]
    print("Embedding chunks (this may take a moment)...", file=sys.stderr)
    embed_model = SentenceTransformer(model_name)
    vecs = embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
    vecs = vecs.astype("float32")
    d = vecs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(vecs)
    FAISS_INDEX = index
    CHUNK_VECTORS = vecs
    print("FAISS index built with", FAISS_INDEX.ntotal, "vectors.", file=sys.stderr)
    return embed_model

def retrieve_top_k(query: str, embed_model: SentenceTransformer, top_k: int = TOP_K_DEFAULT) -> List[Dict[str, Any]]:
    if FAISS_INDEX is None:
        raise RuntimeError("FAISS index not built")
    qv = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    qv = qv.astype("float32")
    D, I = FAISS_INDEX.search(qv, top_k)
    idxs = I[0].tolist()
    results = []
    for idx in idxs:
        if idx < 0 or idx >= len(CHUNK_META):
            continue
        results.append(CHUNK_META[idx])
    return results

# ---------------- OpenAI JSON call helper ----------------
def call_openai_json(prompt: str) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a JSON-only responder. Return only JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
    )
    try:
        content = resp["choices"][0]["message"]["content"]
    except Exception:
        content = resp.choices[0].message.content
    txt = content.strip()
    start_idx = txt.find("{")
    if start_idx == -1:
        raise ValueError("OpenAI returned no JSON object:\n" + txt[:800])
    brace = 0
    end_idx = None
    for i in range(start_idx, len(txt)):
        if txt[i] == "{":
            brace += 1
        elif txt[i] == "}":
            brace -= 1
            if brace == 0:
                end_idx = i + 1
                break
    json_str = txt[start_idx:end_idx] if end_idx else txt
    try:
        return json.loads(json_str)
    except Exception:
        fixed = re.sub(r",\s*}", "}", json_str)
        fixed = re.sub(r",\s*\]", "]", fixed)
        return json.loads(fixed)

# ---------------- Few-shot examples & templates (same as notebook) ----------------
FEWSHOT_EXAMPLE_MCQ = """
{"type":"mcq","stem":"What is the primary purpose of L1 regularization?","options":["Increase model capacity","Reduce overfitting by feature selection","Speed up training","Normalize inputs"],"answer":"Reduce overfitting by feature selection","difficulty":"easy","confidence":0.95}
"""

FEWSHOT_EXAMPLE_YESNO = """
{"type":"yesno","stem":"Does L2 regularization drive weights to zero?","answer":"No","difficulty":"easy","confidence":0.9}
"""

FEWSHOT_EXAMPLE_CODE = """
{"type":"code","stem":"Implement L2 penalty function","description":"Write a Python function that computes L2 penalty for a weight vector.","template":"def l2_penalty(w):\\n    # TODO","tests":"import numpy as np\\ndef test_l2():\\n    assert abs(l2_penalty(np.array([1.0,2.0])) - 5.0) < 1e-6","difficulty":"medium","confidence":0.9}
"""

FEWSHOT_EXAMPLE_DESC = """
{"type":"descriptive","stem":"Explain what L1 regularization does to model weights.","reference_answer":"L1 regularization adds the absolute value of weights as a penalty to the loss, encouraging sparsity and pushing some weights to zero, which can perform feature selection.","difficulty":"medium","confidence":0.9}
"""

MCQ_PROMPT_TEMPLATE = """
You are a strict exam question generator. Use ONLY the CONTEXT blocks to generate EXACTLY {n} multiple-choice questions. Provide distribution: easy={ne}, medium={nm}, hard={nh}.
Return ONLY valid JSON (no explanation) with shape:

{{"questions":[{{"type":"mcq","stem":"...","options":["opt1","opt2","opt3","opt4"],"answer":"optX","difficulty":"easy|medium|hard","confidence":0.0}}, ...]}}

Few-shot example:
{fewshot_example}

Context:
{context}
"""

YESNO_PROMPT_TEMPLATE = """
You are a strict exam question generator. Use ONLY the CONTEXT blocks to generate EXACTLY {n} yes/no questions. Provide distribution: easy={ne}, medium={nm}, hard={nh}.
Return ONLY valid JSON with shape:

{{"questions":[{{"type":"yesno","stem":"...","answer":"Yes" or "No","difficulty":"easy|medium|hard","confidence":0.0}}, ...]}}

Few-shot example:
{fewshot_example}

Context:
{context}
"""

CODE_PROMPT_TEMPLATE = """
You are an exam question generator for coding tasks. Use ONLY the CONTEXT blocks to generate EXACTLY {n} coding questions. Provide distribution: easy={ne}, medium={nm}, hard={nh}.
Return ONLY valid JSON:

{{"questions":[{{"type":"code","stem":"short title","description":"one-paragraph","template":"# starter code","tests":"<PYTEST MODULE TEXT OR JSON CASES>","difficulty":"easy|medium|hard","confidence":0.0}}, ...]}}

Few-shot example:
{fewshot_example}

Context:
{context}
"""

DESCRIPTIVE_PROMPT_TEMPLATE = """
You are an exam question generator. Use ONLY the CONTEXT blocks to generate EXACTLY {n} descriptive questions. Provide distribution: easy={ne}, medium={nm}, hard={nh}.
For each question include 'reference_answer' (2-4 sentence ideal answer).
Return ONLY valid JSON:

{{"questions":[{{"type":"descriptive","stem":"...","reference_answer":"...","difficulty":"easy|medium|hard","confidence":0.0}}, ...]}}

Few-shot example:
{fewshot_example}

Context:
{context}
"""

# ---------------- Difficulty helpers ----------------
def difficulty_split_counts(n: int) -> Tuple[int,int,int]:
    if n <= 0:
        return 0,0,0
    r_e, r_m, r_h = DIFF_RATIOS
    raw_e = n * r_e
    raw_m = n * r_m
    e = int(math.floor(raw_e))
    m = int(math.floor(raw_m))
    h = int(math.floor(n - raw_e - raw_m))
    assigned = e + m + h
    remainder = n - assigned
    for idx in range(remainder):
        if idx == 0:
            e += 1
        elif idx == 1:
            m += 1
        else:
            h += 1
    return e, m, h

# ---------------- Parse custom proportions (list input handler) ----------------
def interpret_custom_list(custom_list: List[Any], n_total: int) -> Tuple[int,int,int,int]:
    """
    Accept custom_list as [a,b,c,d] where a..d may be ints or floats (fractions or percentages).
    Returns integer counts summing to n_total.
    """
    if not isinstance(custom_list, (list, tuple)) or len(custom_list) != 4:
        raise ValueError("customProportions must be a list of 4 numbers")
    nums = []
    for v in custom_list:
        if isinstance(v, str):
            try:
                v = float(v.strip())
            except:
                raise ValueError("customProportions contains non-numeric string")
        if v is None:
            raise ValueError("customProportions contains None")
        nums.append(float(v))
    # If all are integers and sum exactly n_total -> absolute counts
    if all(float(x).is_integer() for x in nums) and int(sum(nums)) == n_total:
        return tuple(int(x) for x in nums)
    # treat as proportions: if sum > 1 treat as percentages
    s = sum(nums)
    if s <= 0:
        raise ValueError("customProportions sum must be > 0")
    if s > 1.0 and s <= 100.0 + 1e-6:
        props = [x / 100.0 for x in nums]
    else:
        props = [(x if x <= 1.0 else x / 100.0) for x in nums]
    tot = sum(props)
    props = [p / tot for p in props]
    raw_counts = [n_total * p for p in props]
    floored = [int(math.floor(x)) for x in raw_counts]
    assigned = sum(floored)
    remainder = n_total - assigned
    fracs = sorted([(raw_counts[i] - floored[i], i) for i in range(4)], reverse=True)
    i = 0
    while remainder > 0:
        idx = fracs[i % 4][1]
        floored[idx] += 1
        remainder -= 1
        i += 1
    return tuple(floored)

# ---------------- RAG generation (uses your original functions) ----------------
def make_context_text(retrieved_chunks: List[Dict[str, Any]]) -> str:
    parts = []
    for i, ch in enumerate(retrieved_chunks, start=1):
        meta = ch.get("meta", {})
        tag = f"{meta.get('path','')}:chunk_{ch.get('id')}"
        parts.append(f"--- CONTEXT {i} ({tag}) ---\n{ch['text']}\n")
    return "\n".join(parts)

def generate_candidates_with_rag_difficulty(chunks_meta: List[Dict[str, Any]], embed_model: SentenceTransformer,
                                            target: Dict[str,int], top_k: int = TOP_K_DEFAULT) -> List[Dict[str, Any]]:
    candidates = []
    def gen_for_type(prompt_template, qtype, total_needed, fewshot_example):
        nonlocal candidates
        if total_needed <= 0:
            return 0
        created = 0
        for cm in chunks_meta:
            if created >= total_needed:
                break
            kws = cm.get("keywords", [])
            kw_text = " ".join(kws[:6])
            query_text = f"{kw_text} {qtype} question"
            try:
                retrieved = retrieve_top_k(query_text, embed_model, top_k=top_k)
            except Exception:
                retrieved = []
            context = make_context_text(retrieved)
            remaining = total_needed - created
            batch = min(3, remaining) if qtype in ("mcq","yesno") else 1
            be_total, bm_total, bh_total = difficulty_split_counts(remaining)
            be = min(be_total, batch)
            bm = min(bm_total, max(0, batch - be))
            bh = max(0, batch - be - bm)
            prompt = prompt_template.format(n=batch, ne=be, nm=bm, nh=bh, context=context, fewshot_example=fewshot_example)
            try:
                j = call_openai_json(prompt)
                qs = j.get("questions", [])
                for q in qs:
                    if "difficulty" not in q or q.get("difficulty") not in ("easy","medium","hard"):
                        q["difficulty"] = "medium"
                    q["id"] = str(uuid.uuid4())
                    q["type"] = qtype
                    q["source_chunk"] = cm.get("id")
                    q["meta_context_ids"] = [r.get("id") for r in retrieved]
                    candidates.append(q)
                created += len(qs)
            except Exception as e:
                print(f"[Generation error type={qtype} chunk={cm.get('id')}] {e}", file=sys.stderr)
        return created

    gen_for_type(MCQ_PROMPT_TEMPLATE, "mcq", target.get("mcq",0), FEWSHOT_EXAMPLE_MCQ)
    gen_for_type(YESNO_PROMPT_TEMPLATE, "yesno", target.get("yesno",0), FEWSHOT_EXAMPLE_YESNO)
    gen_for_type(CODE_PROMPT_TEMPLATE, "code", target.get("code",0), FEWSHOT_EXAMPLE_CODE)
    gen_for_type(DESCRIPTIVE_PROMPT_TEMPLATE, "descriptive", target.get("descriptive",0), FEWSHOT_EXAMPLE_DESC)
    return candidates

# ---------------- Deduplication & assembly ----------------
def deduplicate_questions(candidates: List[Dict[str, Any]], embed_model: SentenceTransformer, threshold: float = DEDUP_SIM_THRESHOLD) -> List[Dict[str, Any]]:
    if not candidates:
        return []
    stems = [ (c.get("stem","")[:400]) for c in candidates ]
    vecs = embed_model.encode(stems, convert_to_numpy=True, normalize_embeddings=True)
    kept = []
    used = np.zeros(len(candidates), dtype=bool)
    def score(i):
        conf = candidates[i].get("confidence")
        try:
            return float(conf) if conf is not None else 0.0
        except Exception:
            return 0.0
    order = sorted(range(len(candidates)), key=lambda i: -score(i))
    for i in order:
        if used[i]:
            continue
        kept.append(candidates[i])
        vi = vecs[i]
        sims = (vecs @ vi).astype(float)
        similar_idx = np.where(sims >= threshold)[0]
        for j in similar_idx:
            used[j] = True
    return kept

def validate_question(q: Dict[str, Any]) -> Tuple[bool, str]:
    typ = q.get("type")
    if typ == "mcq":
        if not q.get("stem") or not q.get("options") or not q.get("answer"):
            return False, "Missing mcq fields"
        if not isinstance(q.get("options"), list) or len(q.get("options")) != 4:
            return False, "MCQ must have 4 options"
        if q["answer"] not in q["options"]:
            return False, "Answer not in options"
    elif typ == "yesno":
        if q.get("answer") not in ("Yes", "No"):
            return False, "YesNo answer invalid"
    elif typ == "code":
        if not q.get("tests"):
            return False, "Code missing tests"
    elif typ == "descriptive":
        if not q.get("reference_answer"):
            return False, "Descriptive missing reference_answer"
    else:
        return False, f"Unknown type: {typ}"
    return True, "OK"

def assemble_test_from_candidates_with_difficulty(candidates: List[Dict[str, Any]], n_mcq: int, n_yesno: int, n_code: int, n_desc: int) -> Dict[str, Any]:
    bucket = candidates
    needs = {"mcq": n_mcq, "yesno": n_yesno, "code": n_code, "descriptive": n_desc}
    selected = []
    for diff in ("easy","medium","hard"):
        for qtype in ("mcq","yesno","code","descriptive"):
            need = needs[qtype]
            if need <= 0:
                continue
            chosen = [q for q in bucket if q.get("type")==qtype and q.get("difficulty")==diff][:need]
            selected.extend(chosen)
            needs[qtype] -= len(chosen)
    valid_questions = []
    answer_key = {}
    for q in selected:
        ok, reason = validate_question(q)
        q["_valid"] = ok
        q["_validation_reason"] = reason
        valid_questions.append(q)
        if q.get("type") in ("mcq","yesno"):
            answer_key[q["id"]] = q.get("answer")
        else:
            answer_key[q["id"]] = None
    def diff_order_value(q):
        return {"easy":0,"medium":1,"hard":2}.get(q.get("difficulty","medium"),1)
    valid_questions.sort(key=lambda x: diff_order_value(x))
    test = {
        "test_id": str(uuid.uuid4()),
        "created_at": datetime.utcnow().isoformat()+"Z",
        "question_count": len(valid_questions),
        "questions": valid_questions,
        "answer_key": answer_key
    }
    return test

# ---------------- Grading helpers (unchanged) ----------------
def grade_descriptive_with_openai(reference_answer: str, student_answer: str, max_tokens: int = 300) -> Dict[str, Any]:
    prompt = f"""
You are an objective grader. Compare the STUDENT_ANSWER below to the REFERENCE_ANSWER.
Return ONLY valid JSON with keys: score (0-100 integer), feedback (short 1-3 sentence suggestion), confidence (0.0-1.0 float).

REFERENCE_ANSWER:
\"\"\"{reference_answer}\"\"\" 

STUDENT_ANSWER:
\"\"\"{student_answer}\"\"\" 

Scoring rules:
- 90-100: captures nearly all key points.
- 70-89: captures most points; minor omissions.
- 40-69: partial.
- 0-39: incorrect or irrelevant.

Return concise JSON.
"""
    try:
        j = call_openai_json(prompt)
    except Exception:
        return {"score": 0, "feedback": "Auto-grader failed; review manually.", "confidence": 0.0}
    score = j.get("score") if isinstance(j, dict) else None
    feedback = j.get("feedback") if isinstance(j, dict) else None
    confidence = j.get("confidence") if isinstance(j, dict) else None
    try:
        score = int(score) if score is not None else None
    except Exception:
        score = None
    try:
        confidence = float(confidence) if confidence is not None else None
    except Exception:
        confidence = None
    return {"score": score, "feedback": feedback, "confidence": confidence}

def grade_submission_basic(test: Dict[str, Any], submission: Dict[str, Any]) -> Dict[str, Any]:
    total = 0; correct = 0; details = {}
    for q in test["questions"]:
        total += 1
        qid = q["id"]; typ = q["type"]
        correct_ans = test["answer_key"].get(qid)
        student_ans = submission.get(qid)
        ok = False; note = ""; score = None; feedback = None
        if typ in ("mcq","yesno"):
            ok = (student_ans == correct_ans)
            score = 100 if ok else 0
        elif typ == "code":
            ok = False; note = "Code not auto-graded in prototype"; score = None
        elif typ == "descriptive":
            ok = None; score = None
        details[qid] = {"question_type": typ, "correct_answer": correct_ans, "submitted": student_ans, "correct": ok, "score": score, "note": note, "feedback": feedback}
        if ok is True:
            correct += 1
    return {"correct": correct, "total": total, "details": details}

# ---------------- Mixed helpers ----------------
def distribute_counts(n: int) -> Tuple[int,int,int,int]:
    base = n // 4
    rem = n % 4
    mcq = yes = code = desc = base
    order = ['mcq','yes','code','desc']
    for i in range(rem):
        if order[i] == 'mcq': mcq += 1
        elif order[i] == 'yes': yes += 1
        elif order[i] == 'code': code += 1
        elif order[i] == 'desc': desc += 1
    return mcq, yes, code, desc

# ---------------- Primary programmatic entry point ----------------
def generate_questions_from_docs(docPaths: List[str], numQuestions: int, qType: str,
                                 useCustomProportions: bool=False, customProportions: List[Any]=None) -> Dict[str, Any]:
    """
    Programmatic wrapper of your notebook flow. Returns a dict ready to be JSON-serialized
    with keys: ok, sessionId, test_id, questions (list).
    """
    # Validate input
    if numQuestions <= 0:
        return {"ok": False, "error": "numQuestions must be positive integer"}
    # Check files existence
    files = docPaths or []
    missing = [f for f in files if not Path(f).exists()]
    if missing:
        return {"ok": False, "error": f"Missing uploaded file(s): {missing}"}

    # Extract & chunk
    chunks_texts = []
    chunk_meta = []
    chunk_id = 0
    for f in files:
        try:
            txt = extract_text_from_file(f)
        except Exception as e:
            return {"ok": False, "error": f"Error extracting {f}: {e}"}
        cks = chunk_text_fast(txt, chunk_chars=CHUNK_CHARS, overlap=CHUNK_OVERLAP)
        for i, c in enumerate(cks):
            chunk_meta.append({"id": chunk_id, "text": c, "meta": {"path": f, "page_idx": i}})
            chunks_texts.append(c)
            chunk_id += 1

    if not chunks_texts:
        return {"ok": False, "error": "No text extracted from provided files."}

    # TF-IDF keywords
    try:
        keywords_per_chunk = compute_chunk_keywords(chunks_texts, top_k=6)
        for i, kw in enumerate(keywords_per_chunk):
            chunk_meta[i]["keywords"] = kw
    except Exception as e:
        print("TF-IDF failed:", e, file=sys.stderr)
        for i in range(len(chunk_meta)):
            chunk_meta[i]["keywords"] = []

    # Build embeddings & FAISS
    try:
        embed_model = build_embeddings_index(chunk_meta, model_name=EMBED_MODEL_NAME)
    except Exception as e:
        return {"ok": False, "error": f"Embedding/FAISS build failed: {e}"}

    global CHUNK_META
    CHUNK_META = chunk_meta

    # Determine target counts for each type
    n_mcq = n_yesno = n_code = n_desc = 0
    if qType == "combo":
        if useCustomProportions and customProportions:
            try:
                n_mcq, n_yesno, n_code, n_desc = interpret_custom_list(customProportions, numQuestions)
            except Exception as e:
                # fallback to equal distribute
                print(f"Invalid custom proportions: {e}. Falling back to equal split.", file=sys.stderr)
                n_mcq, n_yesno, n_code, n_desc = distribute_counts(numQuestions)
        else:
            n_mcq, n_yesno, n_code, n_desc = distribute_counts(numQuestions)
    else:
        if qType == "mcq":
            n_mcq = numQuestions
        elif qType == "yesno":
            n_yesno = numQuestions
        elif qType == "code":
            n_code = numQuestions
        elif qType == "descriptive":
            n_desc = numQuestions
        else:
            return {"ok": False, "error": f"Unknown qType: {qType}"}

    target = {"mcq": n_mcq, "yesno": n_yesno, "code": n_code, "descriptive": n_desc}

    # Generate candidates
    try:
        candidates = generate_candidates_with_rag_difficulty(CHUNK_META, embed_model, target=target, top_k=TOP_K_DEFAULT)
    except Exception as e:
        return {"ok": False, "error": f"Candidate generation failed: {e}"}

    # Deduplicate semantically
    try:
        deduped = deduplicate_questions(candidates, embed_model, threshold=DEDUP_SIM_THRESHOLD)
    except Exception as e:
        print("Deduplication failed, using raw candidates:", e, file=sys.stderr)
        deduped = candidates

    # Assemble final test
    test = assemble_test_from_candidates_with_difficulty(deduped, n_mcq=n_mcq, n_yesno=n_yesno, n_code=n_code, n_desc=n_desc)

    # Save test + answer key to disk
    ts = now_ts()
    test_path = OUTPUT_DIR / f"test_{ts}.json"
    save_json(test, test_path)
    save_json(test["answer_key"], OUTPUT_DIR / f"answer_key_{ts}.json")

    # Prepare output list trimmed for frontend (do not include answer_key)
    out_questions = []
    for q in test["questions"]:
        out = {}
        out["id"] = q.get("id")
        out["type"] = q.get("type")
        # unify field names: prefer 'stem' if present else try other keys
        out["question"] = q.get("stem") or q.get("question") or q.get("title") or ""
        if q.get("type") == "mcq":
            out["options"] = q.get("options", [])
        elif q.get("type") == "yesno":
            out["options"] = ["Yes", "No"]
        elif q.get("type") == "code":
            out["template"] = q.get("template")
            out["description"] = q.get("description")
            out["tests"] = q.get("tests")
        elif q.get("type") == "descriptive":
            out["reference_answer"] = q.get("reference_answer")
        out["difficulty"] = q.get("difficulty")
        out["metadata"] = {"source_chunk": q.get("source_chunk")}
        out_questions.append(out)

    sessionId = "sess-" + uuid.uuid4().hex[:12]
    result = {"ok": True, "sessionId": sessionId, "test_id": test.get("test_id"), "questions": out_questions, "saved_test_path": str(test_path)}
    return result




# ----------------Answer Report----------------------
def generate_detailed_report(test: Dict[str, Any],
                             submission: Dict[str, Any],
                             evaluation: Dict[str, Any] = None,
                             descriptive_pass_threshold: int = 70,
                             save_path: Path | None = None) -> str:
    """
    Create a detailed per-question textual evaluation in the requested format.
    - `test`: the test dict produced by your pipeline (contains `questions` and `answer_key`).
    - `submission`: mapping question_id -> student's answer (strings).
    - `evaluation`: optional result from evaluate_submission(test, submission). If provided,
      the function will use its descriptive feedback/score instead of re-calling the grader.
    - `descriptive_pass_threshold`: used only if evaluation not provided and we grade descriptives here.
    - `save_path`: optional Path to write the plain-text report. If provided, file is written and path returned.

    Returns the generated multi-line report string.
    """
    lines = []
    lines.append("DETAILED PER-QUESTION EVALUATION")
    lines.append("=" * 40)
    per_eval = evaluation.get("per_question") if (evaluation and "per_question" in evaluation) else None

    for idx, q in enumerate(test.get("questions", []), start=1):
        qid = q.get("id")
        qtype = q.get("type")
        stem = q.get("stem", "").strip()
        lines.append(f"\nQuestion {idx}: {stem}")
        lines.append("-" * 40)

        # Student's submitted answer (may be None)
        student_ans = submission.get(qid)
        # Model/correct answer from answer_key (for objective types)
        correct_ans = test.get("answer_key", {}).get(qid)

        if qtype == "mcq":
            # Show model answer (full option text) and the student's reply
            lines.append(f"Model answer: {correct_ans}")
            lines.append(f"Your reply: {student_ans if student_ans is not None else ''}")
            # Decide correctness (prefer evaluation data if present)
            correct_flag = None
            mark = None
            if per_eval and qid in per_eval:
                entry = per_eval[qid]
                mark = entry.get("mark")
                correct_flag = entry.get("mark") == 1
            else:
                # fallback exact-match
                correct_flag = (student_ans == correct_ans)
                mark = 1 if correct_flag else 0
            lines.append(f"Is correct: {'Yes' if correct_flag else 'No'}")
            lines.append(f"Marks: {mark}")
        elif qtype == "yesno":
            lines.append(f"Model answer: {correct_ans}")
            lines.append(f"Your reply: {student_ans if student_ans is not None else ''}")
            correct_flag = None
            mark = None
            if per_eval and qid in per_eval:
                entry = per_eval[qid]
                mark = entry.get("mark")
                correct_flag = entry.get("mark") == 1
            else:
                correct_flag = (str(student_ans).strip().lower() == str(correct_ans).strip().lower())
                mark = 1 if correct_flag else 0
            lines.append(f"Is correct: {'Yes' if correct_flag else 'No'}")
            lines.append(f"Marks: {mark}")
        elif qtype == "descriptive":
            # Model answer (reference) and student's reply
            model_answer = q.get("reference_answer", "")
            lines.append("Answer:")
            if model_answer:
                for line in model_answer.splitlines():
                    lines.append("  " + line)
            else:
                lines.append("  (no reference answer available)")

            lines.append("Your reply:")
            if student_ans:
                for line in str(student_ans).splitlines():
                    lines.append("  " + line)
            else:
                lines.append("  (no answer submitted)")

            # Analysis & score: prefer evaluation info if present
            score = None; feedback = None; conf = None; passed = None
            if per_eval and qid in per_eval:
                entry = per_eval[qid]
                score = entry.get("score_percent")
                feedback = entry.get("feedback")
                conf = entry.get("confidence")
                passed = entry.get("pass")
            else:
                # call grader directly (this uses your grade_descriptive_with_openai function)
                ref = model_answer or ""
                stu = student_ans or ""
                try:
                    g = grade_descriptive_with_openai(ref, stu)
                    score = g.get("score")
                    feedback = g.get("feedback")
                    conf = g.get("confidence")
                    passed = (score is not None and score >= descriptive_pass_threshold)
                except Exception as e:
                    feedback = f"Auto-grader failed: {e}"
            lines.append("Analysis:")
            if feedback:
                # one- or two-line summary
                for fl in str(feedback).splitlines():
                    lines.append("  " + fl.strip())
            else:
                lines.append("  (no feedback)")
            lines.append(f"Score: {score if score is not None else 'N/A'}")
            lines.append(f"Pass: {'Yes' if passed else 'No'}")
        elif qtype == "code":
            lines.append("Note: Code question — will evaluate soon and get back to you.")
            lines.append("Your submission (saved):")
            if student_ans:
                for line in str(student_ans).splitlines():
                    lines.append("  " + line)
            else:
                lines.append("  (no code submitted)")
        else:
            lines.append(f"Unknown question type '{qtype}' - stored submission:")
            lines.append(str(student_ans))

        lines.append("")  # blank line after each question

    report_text = "\n".join(lines)

    # Optionally save to file
    if save_path:
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(report_text)
        except Exception as e:
            print("Warning: failed to save detailed report:", e)

    return report_text



# ---------------- Main CLI handling ----------------
def main():
    # Read stdin JSON
    try:
        raw = sys.stdin.read()
        if not raw:
            print(json.dumps({"ok": False, "error": "No input JSON provided to generator.py"}))
            return
        params = json.loads(raw)
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"Invalid input JSON: {e}"}))
        return

    docPaths = params.get("docPaths", []) or []
    numQuestions = int(params.get("numQuestions", 0) or 0)
    qType = params.get("qType", "mcq")
    useCustomProportions = bool(params.get("useCustomProportions", False))
    customProportions = params.get("customProportions", None)

    try:
        out = generate_questions_from_docs(docPaths, numQuestions, qType, useCustomProportions, customProportions)
        # Print JSON to stdout for Node to capture
        sys.stdout.write(json.dumps(out))
    except Exception as e:
        sys.stdout.write(json.dumps({"ok": False, "error": str(e)}))

if __name__ == "__main__":
    main()
