"""
Agent 1 자동 평가 러너 (eval_agent1_gold.json 기준)

실행:
  python eval/eval_runner.py
  python eval/eval_runner.py --case EVAL_A1_001   # 단일 케이스

결과:
  eval/eval_results_agent1.json   (상세 점수)
  콘솔에 요약 테이블 출력
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openai import OpenAI
from agents.question_refine_agent import QuestionRefineAgent

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

EVAL_DIR  = ROOT / "eval"
GOLD_PATH = EVAL_DIR / "eval_agent1_gold.json"
OUT_PATH  = EVAL_DIR / "eval_results_agent1.json"

# ─────────────────────────────────────────
# 가중치 (gold JSON과 동일)
# ─────────────────────────────────────────
WEIGHTS = {
    "question_units":     0.25,
    "current_bottleneck": 0.20,
    "hard_case_flags":    0.20,
    "routing":            0.15,
    "taxonomy_tags":      0.10,
    "refined_question":   0.05,
    "privacy_safe":       0.05,
}

_PII_PATTERNS = [
    r"\b\S+대학교\b", r"\b\S+대\s*\d+학년\b", r"\b\d{2,4}학번\b",
    r"\b01[016789]-?\d{3,4}-?\d{4}\b",
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
]


# ─────────────────────────────────────────
# 채점 함수
# ─────────────────────────────────────────

def score_bottleneck(pred: dict, gold: dict) -> float:
    """current_bottleneck 완전 일치 = 1.0, 아니면 0.0"""
    p = pred.get("current_bottleneck", "").strip()
    g = gold.get("current_bottleneck", "").strip()
    return 1.0 if p == g else 0.0


def score_question_units(pred: dict, gold: dict) -> float:
    """
    Gold question_units 각각에 대해 predicted units 중 키워드 overlap이
    있으면 matched로 처리. score = matched / total_gold.
    매우 짧은 gold(<2개) 는 완전일치 위주로 처리.
    """
    gold_units = [u.get("unit", "") for u in gold.get("question_units", [])]
    pred_units = [u.get("unit", "") for u in pred.get("question_units", [])]

    if not gold_units:
        return 1.0
    if not pred_units:
        return 0.0

    def keywords(text: str) -> set:
        # 2글자 이상 토큰
        return {t for t in re.split(r"[\s·,/·\-\(\)]+", text) if len(t) >= 2}

    matched = 0
    for gu in gold_units:
        gk = keywords(gu)
        for pu in pred_units:
            pk = keywords(pu)
            overlap = gk & pk
            if len(overlap) >= max(1, len(gk) // 2):
                matched += 1
                break

    return round(matched / len(gold_units), 3)


def score_hard_case_flags(pred: dict, gold: dict) -> float:
    """
    requires_artifact_review, recency_sensitive, scope_too_broad 각 1/3 가중치.
    risk_flags는 gold에 있으면 pred에도 있어야 함 (포함 여부).
    """
    gold_hf = gold.get("hard_case_flags", {})
    pred_hf = pred.get("hard_case_flags", {})

    scores = []
    for flag in ("requires_artifact_review", "recency_sensitive", "scope_too_broad"):
        gv = bool(gold_hf.get(flag, False))
        pv = bool(pred_hf.get(flag, False))
        scores.append(1.0 if gv == pv else 0.0)

    # risk_flags 포함 여부
    gold_risks = set(gold_hf.get("risk_flags", []))
    pred_risks = set(pred_hf.get("risk_flags", []))
    if gold_risks:
        hit = len(gold_risks & pred_risks) / len(gold_risks)
        scores.append(hit)
    else:
        scores.append(1.0)

    return round(sum(scores) / len(scores), 3)


def score_routing(pred: dict, gold: dict) -> float:
    """
    search_strategy_hint, question_structure, target_role_specificity 각 1/3.
    """
    gold_rh = gold.get("routing_hints", {})
    pred_rh = pred.get("routing_hints", {})

    fields = [
        ("search_strategy_hint",   gold_rh.get("search_strategy_hint",   "")),
        ("question_structure",     gold_rh.get("question_structure",     "")),
        ("target_role_specificity",gold_rh.get("target_role_specificity","")),
    ]
    scores = []
    for field, gv in fields:
        if not gv:          # gold에 없는 필드는 스킵
            continue
        pv = pred_rh.get(field, "")
        scores.append(1.0 if pv == gv else 0.0)

    return round(sum(scores) / len(scores), 3) if scores else 1.0


def score_taxonomy(pred: dict, gold: dict) -> float:
    """
    domain_tags F1 (gold ∩ pred / gold).
    role_tags, question_type_tags도 같은 방식, 평균.
    """
    gold_tt = gold.get("taxonomy_tags", {})
    pred_tt = pred.get("taxonomy_tags", {})

    sub_scores = []
    for tag_key in ("domain_tags", "role_tags", "question_type_tags", "career_stage_tags"):
        gset = set(gold_tt.get(tag_key, []))
        pset = set(pred_tt.get(tag_key, []))
        if not gset:
            continue
        recall = len(gset & pset) / len(gset)
        sub_scores.append(recall)

    return round(sum(sub_scores) / len(sub_scores), 3) if sub_scores else 1.0


def score_refined_question(pred: dict, gold: dict) -> float:
    """
    LLM-as-judge: refined_question이 gold와 방향이 같은지 GPT-4o-mini로 판단.
    1.0 / 0.5 / 0.0 반환.
    """
    gq = gold.get("refined_question", "")
    pq = pred.get("refined_question", "")
    if not pq:
        return 0.0

    prompt = f"""아래 두 진로 멘토링 정제 질문을 비교해.
Gold(정답): {gq}
Pred(예측): {pq}

판단 기준:
- 1.0: 핵심 의도·직무·원하는 도움이 gold와 동일하거나 더 좋음
- 0.5: 방향은 맞지만 중요한 요소(직무명/원하는 도움 종류 등) 일부 누락
- 0.0: 의도가 다르거나 핵심 정보가 빠짐

숫자(1.0 또는 0.5 또는 0.0)만 반환. 설명 없음."""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=5,
    )
    raw = resp.choices[0].message.content.strip()
    try:
        val = float(raw)
        return round(max(0.0, min(1.0, val)), 1)
    except ValueError:
        return 0.5


def score_privacy(pred: dict) -> float:
    """safe_context, search_query에 PII 패턴 없으면 1.0."""
    texts = [
        pred.get("safe_context", ""),
        pred.get("search_query", ""),
        pred.get("refined_question", ""),
    ]
    combined = " ".join(t for t in texts if t)
    for pat in _PII_PATTERNS:
        if re.search(pat, combined):
            return 0.0
    return 1.0


def score_case(pred: dict, gold: dict) -> dict:
    """케이스 하나의 전체 채점 결과 반환."""
    sub = {
        "question_units":     score_question_units(pred, gold),
        "current_bottleneck": score_bottleneck(pred, gold),
        "hard_case_flags":    score_hard_case_flags(pred, gold),
        "routing":            score_routing(pred, gold),
        "taxonomy_tags":      score_taxonomy(pred, gold),
        "refined_question":   score_refined_question(pred, gold),
        "privacy_safe":       score_privacy(pred),
    }
    total = sum(sub[k] * WEIGHTS[k] for k in WEIGHTS)
    return {"sub_scores": sub, "total": round(total, 4)}


# ─────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────

def run_eval(target_case: str | None = None) -> None:
    gold_data = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    cases = gold_data["cases"]

    if target_case:
        cases = [c for c in cases if c["case_id"] == target_case]
        if not cases:
            print(f"[오류] {target_case} 케이스를 찾을 수 없음")
            return

    results = []
    print(f"\n{'='*65}")
    print(f"  Agent 1 평가 시작 — {len(cases)}개 케이스")
    print(f"{'='*65}\n")

    for i, case in enumerate(cases, 1):
        cid      = case["case_id"]
        category = case["category"]
        question = case["original_question"]
        gold     = case["gold"]

        print(f"[{i:02d}/{len(cases)}] {cid} | {category}")
        print(f"  질문: {question[:60]}{'...' if len(question)>60 else ''}")

        try:
            agent = QuestionRefineAgent()
            pred  = agent.analyze(question)
            scores = score_case(pred, gold)

            # 세부 점수 출력
            sub = scores["sub_scores"]
            print(f"  bottleneck:{sub['current_bottleneck']:.1f} "
                  f"units:{sub['question_units']:.2f} "
                  f"flags:{sub['hard_case_flags']:.2f} "
                  f"routing:{sub['routing']:.2f} "
                  f"taxonomy:{sub['taxonomy_tags']:.2f} "
                  f"→ 합계: {scores['total']:.3f}")

            results.append({
                "case_id":         cid,
                "category":        category,
                "original_question": question,
                "scores":          scores,
                "pred_bottleneck": pred.get("current_bottleneck", ""),
                "gold_bottleneck": gold.get("current_bottleneck", ""),
                "pred_question_units": [u.get("unit","") for u in pred.get("question_units",[])],
                "gold_question_units": [u.get("unit","") for u in gold.get("question_units",[])],
                "pred_routing_hints":  pred.get("routing_hints", {}),
                "gold_routing_hints":  gold.get("routing_hints", {}),
                "pred_hard_case_flags": pred.get("hard_case_flags", {}),
                "gold_hard_case_flags": gold.get("hard_case_flags", {}),
            })

        except Exception as e:
            print(f"  [ERROR] {e}")
            results.append({
                "case_id": cid,
                "category": category,
                "original_question": question,
                "scores": {"sub_scores": {}, "total": 0.0},
                "error": str(e),
            })

        print()

    # ── 요약 테이블 ──
    valid = [r for r in results if "error" not in r]
    if valid:
        print(f"\n{'─'*65}")
        print(f"{'케이스':<18} {'bottleneck':>10} {'units':>6} {'flags':>6} {'routing':>7} {'taxonomy':>8} {'total':>7}")
        print(f"{'─'*65}")
        for r in valid:
            sub = r["scores"]["sub_scores"]
            print(f"{r['case_id']:<18} "
                  f"{sub.get('current_bottleneck',0):>10.1f} "
                  f"{sub.get('question_units',0):>6.2f} "
                  f"{sub.get('hard_case_flags',0):>6.2f} "
                  f"{sub.get('routing',0):>7.2f} "
                  f"{sub.get('taxonomy_tags',0):>8.2f} "
                  f"{r['scores']['total']:>7.3f}")

        totals   = [r["scores"]["total"] for r in valid]
        avg_tot  = sum(totals) / len(totals)
        avg_sub  = {k: round(sum(r["scores"]["sub_scores"].get(k,0) for r in valid)/len(valid),3)
                    for k in WEIGHTS}
        print(f"{'─'*65}")
        print(f"{'평균':<18} "
              f"{avg_sub['current_bottleneck']:>10.3f} "
              f"{avg_sub['question_units']:>6.3f} "
              f"{avg_sub['hard_case_flags']:>6.3f} "
              f"{avg_sub['routing']:>7.3f} "
              f"{avg_sub['taxonomy_tags']:>8.3f} "
              f"{avg_tot:>7.3f}")
        print(f"{'='*65}\n")

    # ── 결과 저장 ──
    output = {
        "eval_target": "agent1",
        "total_cases": len(cases),
        "evaluated":   len(valid),
        "average_total": round(sum(r["scores"]["total"] for r in valid)/len(valid),4) if valid else 0,
        "average_sub_scores": {k: round(sum(r["scores"]["sub_scores"].get(k,0) for r in valid)/len(valid),3)
                               for k in WEIGHTS} if valid else {},
        "weights": WEIGHTS,
        "cases": results,
    }
    OUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"결과 저장: {OUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, default=None, help="단일 케이스 ID (예: EVAL_A1_001)")
    args = parser.parse_args()
    run_eval(args.case)
