"""
에이전트 2: 검색검증 에이전트 v2

[에이전트 루프]
관찰0: hard_case_flags 선처리
  → requires_artifact_review=True 이면 즉시 멘토 연결 (검색 생략)
관찰1: 전략 판단 (search_first / mentor_first)
  → Agent 1 hint는 strategy_confidence >= 0.7 일 때만 채택
  → complexity·hard_case_flags·risk_flags 기반 threshold_boost 동적 계산
  → recency_sensitive=True 이면 recency 가중치 상향
행동1: 임베딩 검색 + 코드 기반 recency·privacy 선처리
관찰2: 검색 결과 확인
행동2: LLM 검증 (relevance·evidence_sufficiency·situation_fit)
  → _verify() 에 safe_context·current_bottleneck·expected_answer_type·question_units·risk_flags 전달
관찰3: usable_answer_ids 유효성 확인
  → 빈 배열이면 retrieved 전체 폴백 금지 → 멘토 연결
행동3: 답변 생성 (mode별 프롬프트 분리 + stale_notice 삽입)
관찰4: faithfulness 자기검증 + generated answer privacy 검사
  → 2회 미통과 또는 privacy 위험 → 멘토 연결 폴백
출력: verdict·answer·structured_fallback_reason·retrieval_log

핵심 설계 원칙
- LLM은 의미 판단만 (situation_fit, evidence_sufficiency, 답변생성, 검증)
- 수치 계산은 코드 (recency, 가중합산, pass 판정, threshold_boost 계산)
- 오류 시 항상 멘토 연결 폴백 (과잉 연결 > 과잉 직접답변)
- hard_case_flags → threshold·분기에 반영
- 구조화된 fallback_reason → Agent 3 멘토 매칭 품질 향상
- retrieval_log → Agent 4 평가·개선에 활용
"""

import os
import re
import json
from datetime import datetime
from openai import OpenAI
from db.json_db import get_assetized_answers, update_session
from utils.embedding import get_embedding, top_k_similar

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# ─────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────

def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true"
    return default


def normalize_ids(raw_ids, max_len: int) -> list[int]:
    """
    LLM 반환 usable_answer_ids를 안전한 정수 리스트로 정규화.
    - 문자열·float 허용 (int 변환)
    - 범위 [1, max_len] 벗어나면 제거
    - 빈 리스트 그대로 반환 (호출부에서 멘토 폴백 처리)
    """
    if not isinstance(raw_ids, list):
        return []
    result = []
    for v in raw_ids:
        try:
            i = int(v)
            if 1 <= i <= max_len:
                result.append(i)
        except (TypeError, ValueError):
            continue
    return result


# 개인정보 정규식 선처리 (LLM 호출 전 1차 필터, 생성 답변에도 재사용)
_PRIVACY_PATTERNS = [
    r'\d{2,4}학번',
    r'\d{2,3}-\d{3,4}-\d{4}',
    r'[\w.+-]+@[\w-]+\.[a-z]+',
    r'(?:고려|연세|서울|성균관|한양|이화|숙명|중앙|건국|경희)대.{0,5}\d{2}학번',
]

def _has_privacy_risk(text: str) -> bool:
    """정규식 기반 개인정보 위험 탐지 (검색 결과·생성 답변 공통 사용)"""
    for pattern in _PRIVACY_PATTERNS:
        if re.search(pattern, text):
            return True
    return False


def _calc_recency(answer: dict) -> float:
    """created_at 메타데이터 기반 최신성 점수 계산 (LLM 불필요)"""
    created_at = answer.get("created_at", "")
    if not created_at:
        return 0.5
    try:
        created = datetime.strptime(created_at, "%Y-%m-%d %H:%M")
        age_days = (datetime.now() - created).days
        if age_days <= 180:   return 1.0
        elif age_days <= 365: return 0.8
        elif age_days <= 730: return 0.6
        else:                 return 0.4
    except Exception:
        return 0.5


def _build_stale_notice(stale_ids: list[int]) -> str:
    """답변 생성 프롬프트에 삽입할 stale 경고문"""
    if not stale_ids:
        return ""
    ids_str = ", ".join(str(i) for i in stale_ids)
    return (
        f"\n[주의] 답변 {ids_str}번은 내용은 관련 있으나 시점이 오래됐을 수 있습니다. "
        "채용 트렌드·연봉·특정 회사 기준 등 시점 민감 정보를 인용할 경우 "
        "반드시 '최신 정보는 직접 확인이 필요합니다'라고 표시하세요.\n"
    )


# ─────────────────────────────────────────
# 프롬프트
# ─────────────────────────────────────────

COMPLEXITY_PROMPT = """아래 진로 질문을 search_first 또는 mentor_first 중 하나로 분류해줘.
반드시 JSON만 출력해.

[분류 기준]

search_first:
- 답변이 일반적인 직무 정보, 준비 방법, 업계 현황, 면접 방식으로 충분한 질문
- 사용자의 개인 배경이 없거나, 있어도 답변 방향을 크게 바꾸지 않는 질문

예시:
- "데이터 분석가가 되려면 어떤 스킬을 쌓아야 하나요?"
- "컨설팅 면접은 케이스 스터디가 주로 나오나요?"
- "마케팅 신입 연봉 평균이 어느 정도인가요?"
- "UX 디자이너 포트폴리오에 뭘 담아야 하나요?"

mentor_first:
- 전공, 학년, 지역, 경제 상황, 가족 배경, 현재 스펙, 진로 전환 여부가 답변 방향을 바꾸는 핵심 정보인 질문
- "내 상황에서 가능한가?", "나에게 맞는가?", "제 스펙으로 되는가?"처럼 개인 판단이 핵심인 질문
- 애매하면 mentor_first로 분류 (잘못된 직접 답변보다 멘토 연결이 더 안전)

예시:
- "경영학과 3학년인데 개발자로 전향 가능할까요? 지방 거주라 부트캠프 접근도 어려워요"
- "가족 중 대졸자가 없어서 취업 과정을 잘 모르는데, 제 스펙으로 컨설팅 갈 수 있을까요?"
- "인턴 경험이 있는데 이 직무가 나한테 진짜 맞는지 모르겠어요. 제 상황을 봐주세요"
- "비전공자인데 AI 직무 전환하려면 어디서부터 시작해야 할지 막막해요"

[출력 형식]
{{"strategy": "search_first 또는 mentor_first", "complexity": "low 또는 medium 또는 high", "personal_context_strength": "weak 또는 moderate 또는 strong", "strategy_confidence": 0.0, "reason": "판단 이유 한 문장"}}

strategy_confidence: 이 분류가 얼마나 확실한지. 0.9+ = 명확, 0.7~0.8 = 비교적 명확, 0.5~0.6 = 애매, 0.5 미만 = 불명확

질문: {question}"""


VERIFY_PROMPT = """아래 질문에 대해 검색된 기존 멘토 답변들이 AI 답변의 근거로 충분한지 평가해줘.
반드시 JSON만 출력해.

[평가 원칙]
- 검색 답변 전체를 막연히 평균 내지 말고, 실제 답변 근거로 사용할 수 있는 답변이 몇 개인지 판단해라.
- 무관한 답변이 많으면 relevance와 evidence_sufficiency를 낮춰라.
- 애매하면 낮은 점수를 줘라.
- 개인 식별 가능성이 있으면 privacy_safe=false.

[점수 기준 (relevance, evidence_sufficiency, situation_fit 공통)]
0.9~1.0: 거의 동일한 질문/상황에 직접 답할 수 있음
0.7~0.8: 핵심 주제가 같고 답변 근거로 충분히 활용 가능
0.5~0.6: 일부 활용 가능하지만 중요한 맥락이 부족함
0.3~0.4: 주제만 유사하고 직접 근거로 쓰기에 약함
0.0~0.2: 거의 관련 없음

[privacy_safe=false 기준]
- 실명 / 이메일 / 전화번호
- 학교+학번 조합
- 회사+팀+연차 등 개인 식별 가능한 조합
- 희귀한 경력 경로 (특정인 식별 가능)
- 가족·재정·건강 등 민감 개인 맥락

[stale_but_useful_ids 기준]
내용은 관련 있지만 오래됐을 가능성이 있는 답변 번호를 표시해라.
- 직무 역량·준비 방법 자체는 유효하지만, 채용 트렌드·연봉·회사 기준 등 시점 민감 정보가 포함된 경우
- 사용은 가능하지만 "최신 정보는 확인 필요" 주의가 필요한 답변들

[추가 맥락 — 판단에 활용]
멘티 현재 병목: {current_bottleneck}
예상 답변 유형: {expected_answer_type}
세부 질문 단위: {question_units}
위험 플래그: {risk_flags}

[Hard-case 판단 기준]
- 직무 전환 질문: 검색 답변이 source_role → target_role 사이의 bridge를 설명하는가?
  단순히 target_role 준비법만 다루면 situation_fit을 낮게 준다.
- 기술 전이 가능성 질문: 검색 답변이 보유 기술의 타 도메인 적용 가능성을 다루는가?
  일반 직무 정보만 있으면 situation_fit을 낮게 준다.
- question_units 중 answerability="searchable" 인 것만 커버되었다면,
  usable_answer_ids는 해당 답변만 담고, 나머지 unit은 멘토 연결로 처리될 것임을 감안한다.
  → 이 경우 evidence_sufficiency를 지나치게 낮추지 않아도 된다 (partial answer가 적절함).

[출력 형식]
{{"scores": {{"relevance": 0.0, "evidence_sufficiency": 0.0, "situation_fit": 0.0, "privacy_safe": true}}, "usable_answer_ids": [1, 2], "stale_but_useful_ids": [], "privacy_risk_reason": "없음 또는 위험 이유", "reason": "전체 판단 이유 한 문장"}}

질문: {question}
멘티 맥락: {context}
검색된 답변들:
{retrieved_answers}"""


DIRECT_ANSWER_PROMPT = """너는 맨투맨 진로 멘토링 서비스의 AI 상담사야.
아래 [참고 답변들]을 주된 근거로 삼아 멘티에게 답변해.
{stale_notice}
[핵심 원칙]
- 참고 답변에서 확인되는 내용만 근거로 사용한다.
- 참고 답변의 개인 경험은 직접 인용하지 말고, 개인을 식별할 수 없도록 일반화해서 요약한다.
  ("기존 멘토 답변에서는 ~ 경향이 반복적으로 언급됩니다" 형식 권장)
- 참고 답변에 없는 구체적 수치, 회사명, 합격 가능성, 연봉, 개인 적합성은 단정하지 않는다.
- 일반적 업계 상식으로 보충할 경우 반드시 "일반적으로는" 또는 "보통 업계에서는"으로 표시한다.
- 개인정보, 학교+학번, 회사+팀, 실명, 희귀한 경력 조합은 노출하지 않는다.

[답변 구조]
첫 줄: "기존 멘토 경험과 직무 정보를 바탕으로 한 참고 답변입니다."

1. 기존 멘토 답변에서 확인되는 핵심 내용 (비식별화 요약)
2. 멘티 상황에 적용해볼 수 있는 점
3. 일반적으로 보충할 수 있는 준비 방향 (있을 경우, "일반적으로는..." 표시)
4. 지금 할 수 있는 다음 행동 2~3개

마지막 줄: "이 내용이 본인 상황과 다르다면 현직자 멘토 연결을 추천해드릴게요."

질문: {question}
멘티 맥락: {context}
참고 답변들:
{retrieved_answers}"""


PARTIAL_ANSWER_PROMPT = """너는 맨투맨 진로 멘토링 서비스의 AI 상담사야.
아래 참고 답변은 현재 질문과 일부만 관련되어 있어.
따라서 확정 답변이 아니라 참고 가능한 범위만 정리해.
{stale_notice}
[핵심 원칙]
- 참고 답변과 현재 질문이 완전히 일치하지 않는다는 점을 명확히 밝혀라.
- 참고 답변에서 실제로 도움되는 부분만 가져와라.
- 개인 적합성, 합격 가능성, 구체적 진로 판단은 하지 않는다.
- 판단이 어려운 부분은 "이 부분은 기존 답변만으로 판단하기 어렵습니다"라고 말한다.
- 개인정보는 일반화한다.

[답변 구조]
첫 줄: "완전히 일치하는 사례는 없지만 참고할 수 있는 경험을 정리해드립니다."

1. 참고 가능한 내용 (비식별화 요약)
2. 현재 질문에 그대로 적용하기 어려운 부분
3. 일반적으로 생각해볼 수 있는 준비 방향 ("일반적으로는..." 표시 필수)
4. 현직자 멘토에게 확인하면 좋은 질문 2~3개

마지막 줄: "본인 상황에 맞는 더 정확한 조언을 위해 현직자 멘토 연결을 권장드립니다."

질문: {question}
멘티 맥락: {context}
참고 답변들:
{retrieved_answers}"""


FAITHFULNESS_PROMPT = """아래 AI 생성 답변의 품질을 자기검증해줘.
반드시 JSON만 출력해.

[검증 기준]

grounded (0.0~1.0): 답변의 핵심 주장 중 참고 답변에서 실제로 확인되는 비율
  0.9~1.0: 거의 모든 핵심 주장이 참고 답변에 근거함
  0.7~0.8: 대부분 근거가 있으나 일부 일반론이 있음
  0.5~0.6: 절반 정도만 근거가 있음
  0.0~0.4: 참고 답변보다 모델의 일반 지식에 많이 의존함

supplement_marked (true/false):
  참고 답변 외 일반론이 "일반적으로" 등으로 명확히 표시되었는가

factually_sound (true/false):
  명백히 틀린 정보나 과도한 단정이 없는가
  합격 가능성, 구체적 연봉, 취업 보장 등 근거 없는 단정이 있으면 false

privacy_leaked (true/false):
  개인 식별 가능 정보가 노출되었는가

over_generalized (true/false):
  참고 답변보다 일반론/추측에 기대어 결론을 확장했는가
  합격 가능성, 개인 적합성, 연봉, 특정 회사 기준을 근거 없이 단정하면 true

pass: 아래 조건 모두 충족 시 true
  - grounded >= {grounded_threshold}
  - supplement_marked == true 또는 보충 내용 없음
  - factually_sound == true
  - privacy_leaked == false
  - over_generalized == false

[출력 형식]
{{"grounded": 0.0, "supplement_marked": true, "factually_sound": true, "privacy_leaked": false, "over_generalized": false, "pass": true, "reason": "한 줄 판단 이유"}}

참고 답변들:
{retrieved_answers}

AI 생성 답변:
{generated_answer}"""


# ─────────────────────────────────────────
# 에이전트 클래스
# ─────────────────────────────────────────

class SearchVerifyAgent:
    THRESHOLD     = {"search_first": 0.65, "mentor_first": 0.75}
    MID_THRESHOLD = {"search_first": 0.45, "mentor_first": 0.50}
    SIM_THRESHOLD = {"search_first": 0.50, "mentor_first": 0.45}

    # 기본 가중치 (recency는 코드 계산, LLM은 나머지 3개만)
    WEIGHTS = {
        "search_first": {"relevance": 0.30, "evidence_sufficiency": 0.30, "situation_fit": 0.20, "recency": 0.20},
        "mentor_first": {"relevance": 0.15, "evidence_sufficiency": 0.20, "situation_fit": 0.50, "recency": 0.15},
    }

    # mode별 faithfulness grounded 임계값
    GROUNDED_THRESHOLD = {"direct": 0.7, "partial": 0.5}

    # Agent 1 hint를 채택할 최소 confidence
    STRATEGY_CONFIDENCE_MIN = 0.7

    def run(
        self,
        session_id: str,
        refined_question: str,
        conversation_summary: str,
        routing_hints: dict | None = None,
        search_query: str | None = None,
        # ── Agent 1 진단 필드 (v2 신규) ──
        safe_context: str | None = None,
        current_bottleneck: str | None = None,
        expected_answer_type: str | None = None,
        question_units: list[dict] | None = None,
        hard_case_flags: dict | None = None,
    ):
        print("[검색검증 에이전트 v2] 실행 중...")
        routing_hints    = routing_hints or {}
        hard_case_flags  = hard_case_flags or {}
        question_units   = question_units or []
        safe_context     = safe_context or conversation_summary
        current_bottleneck   = current_bottleneck or ""
        expected_answer_type = expected_answer_type or ""

        risk_flags = hard_case_flags.get("risk_flags", [])
        if isinstance(risk_flags, str):
            risk_flags = [risk_flags] if risk_flags else []

        # ── retrieval_log 초기화 (Agent 4 평가용) ──
        retrieval_log: dict = {
            "session_id":      session_id,
            "timestamp":       datetime.now().isoformat(),
            "hard_case_flags": hard_case_flags,
            "question_units":  question_units,
            "strategy":        None,
            "threshold_boost": 0.0,
            "retrieved_count": 0,
            "usable_count":    0,
            "stale_ids":       [],
            "avg_score":       0.0,
            "verdict":         None,
            # v4 신규: 직무전환·전이 가능성 맥락 보존
            "bridge_hypothesis":       hard_case_flags.get("bridge_hypothesis", ""),
            "transferable_skills":     hard_case_flags.get("transferable_skills", []),
            "target_domain_candidates": hard_case_flags.get("target_domain_candidates", []),
        }

        # ─────────────────────────────────────────
        # 관찰0: hard_case_flags 선처리
        # ─────────────────────────────────────────
        if safe_bool(hard_case_flags.get("requires_artifact_review")):
            print("  관찰0 | requires_artifact_review=True → 포트폴리오·이력서 검토 필요 → 멘토 연결")
            retrieval_log["verdict"] = "mentor_needed"
            return self._mentor_fallback(
                session_id, "mentor_first",
                fallback_type="requires_artifact_review",
                reason="포트폴리오·이력서 등 자료 검토가 필요한 질문입니다",
                mentor_match_hints={
                    "desired_help":  "이력서·포트폴리오 피드백",
                    "risk_flags":    risk_flags,
                    "question_units": [u.get("unit", "") for u in question_units],
                },
                retrieval_log=retrieval_log,
            )

        # ─────────────────────────────────────────
        # 관찰1: 전략 판단
        # ─────────────────────────────────────────
        strategy_hint       = routing_hints.get("search_strategy_hint")
        strategy_confidence = safe_float(routing_hints.get("search_strategy_confidence", 0.0))

        if strategy_hint in ("search_first", "mentor_first") and strategy_confidence >= self.STRATEGY_CONFIDENCE_MIN:
            strategy_result = {
                "strategy":                  strategy_hint,
                "complexity":                "medium",
                "personal_context_strength": routing_hints.get("personal_context_strength", "moderate"),
                "strategy_confidence":       strategy_confidence,
                "reason":                    f"Agent 1 hint 채택 (confidence={strategy_confidence:.2f})",
            }
            print(f"  관찰1 | Agent 1 hint 채택: {strategy_hint} (confidence={strategy_confidence:.2f})")
        else:
            if strategy_hint and strategy_confidence < self.STRATEGY_CONFIDENCE_MIN:
                print(f"  관찰1 | Agent 1 hint confidence 부족 ({strategy_confidence:.2f} < {self.STRATEGY_CONFIDENCE_MIN}) → LLM 재판단")
            strategy_result = self._judge_strategy(refined_question)

        strategy   = strategy_result.get("strategy", "mentor_first")
        complexity = strategy_result.get("complexity", "medium")
        if strategy not in self.THRESHOLD:
            strategy = "mentor_first"

        # ── threshold_boost 동적 계산 ──
        threshold_boost = 0.0
        if complexity == "high":
            threshold_boost += 0.05
        if risk_flags:
            threshold_boost += 0.05
        if safe_bool(hard_case_flags.get("scope_too_broad")):
            threshold_boost += 0.03
        if safe_bool(hard_case_flags.get("recency_sensitive")):
            threshold_boost += 0.03

        retrieval_log["strategy"]        = strategy
        retrieval_log["threshold_boost"] = threshold_boost

        print(
            f"  관찰1 | 전략: {strategy} / complexity: {complexity} / "
            f"threshold_boost: +{threshold_boost:.2f} / "
            f"risk_flags: {risk_flags or '없음'}"
        )

        # ── recency_sensitive / recency_level 이면 recency 가중치 상향 ──
        weights = {k: v for k, v in self.WEIGHTS[strategy].items()}  # 복사
        recency_level = hard_case_flags.get("recency_level", "")
        if safe_bool(hard_case_flags.get("recency_sensitive")):
            # recency_level=high면 추가 0.05 더 올림
            extra = 0.15 if recency_level == "high" else 0.10
            weights["recency"] = min(1.0, weights["recency"] + extra)
            other_keys = [k for k in weights if k != "recency"]
            total_other = sum(weights[k] for k in other_keys)
            if total_other > 0:
                scale = (1.0 - weights["recency"]) / total_other
                for k in other_keys:
                    weights[k] *= scale
            print(f"  관찰1 | recency_sensitive=True (level={recency_level or 'n/a'}) → recency 가중치 {weights['recency']:.2f}로 상향")

        # ─────────────────────────────────────────
        # 행동1: 임베딩 검색
        # ─────────────────────────────────────────
        effective_query = search_query or refined_question
        if search_query:
            print("  행동1 | search_query 사용 (Agent 1 최적화 쿼리)")

        answers   = get_assetized_answers()
        retrieved = []
        sim_threshold = self.SIM_THRESHOLD[strategy]

        if answers:
            try:
                query_vec = get_embedding(effective_query)
                top       = top_k_similar(query_vec, answers, vec_field="embedding", k=5)
                retrieved = [item for item, score in top if score >= sim_threshold]
                print(f"  행동1 | 임베딩 검색: {len(retrieved)}개 발견 (유사도 >= {sim_threshold})")
            except Exception as e:
                print(f"  행동1 | 임베딩 검색 실패: {e} → 멘토 연결")
                retrieval_log["verdict"] = "mentor_needed"
                return self._mentor_fallback(
                    session_id, strategy,
                    fallback_type="search_error",
                    reason="임베딩 검색 실패",
                    retrieval_log=retrieval_log,
                )
        else:
            print("  행동1 | 자산 DB 비어있음")

        # ─────────────────────────────────────────
        # 관찰2: 검색 결과 확인
        # ─────────────────────────────────────────
        if not retrieved:
            print("  관찰2 | 유사 답변 없음 → 멘토 연결")
            retrieval_log["verdict"] = "mentor_needed"
            return self._mentor_fallback(
                session_id, strategy,
                fallback_type="no_similar_answers",
                reason="유사한 멘토 답변이 아직 없습니다",
                retrieval_log=retrieval_log,
            )

        retrieval_log["retrieved_count"] = len(retrieved)

        # 코드 기반 recency 선계산
        recency_avg = sum(_calc_recency(a) for a in retrieved) / len(retrieved)

        # 코드 기반 privacy 선처리 (검색 결과)
        combined_text = " ".join(a.get("answer_content", "") for a in retrieved)
        pre_privacy_unsafe = _has_privacy_risk(combined_text)
        if pre_privacy_unsafe:
            print("  행동1 | 정규식 개인정보 위험 감지 (검색 결과) → privacy_safe 강제 false")

        print(
            f"  관찰2 | {len(retrieved)}개 후보 / recency 평균: {recency_avg:.2f} / "
            f"privacy 선처리: {'위험' if pre_privacy_unsafe else '안전'}"
        )

        # ─────────────────────────────────────────
        # 행동2: LLM 검증
        # ─────────────────────────────────────────
        verify_result = self._verify(
            refined_question, safe_context, retrieved,
            current_bottleneck=current_bottleneck,
            expected_answer_type=expected_answer_type,
            question_units=question_units,
            risk_flags=risk_flags,
        )
        scores    = verify_result.get("scores", {})
        raw_ids   = verify_result.get("usable_answer_ids", [])
        stale_ids = verify_result.get("stale_but_useful_ids", [])

        # normalize_ids: 타입 안전 정규화
        usable_ids = normalize_ids(raw_ids, len(retrieved))
        stale_ids  = normalize_ids(stale_ids, len(retrieved))

        # ─────────────────────────────────────────
        # 관찰3: usable_ids 유효성 확인
        # ─────────────────────────────────────────
        if not usable_ids:
            # v2 핵심 변경: 빈 배열이면 retrieved 전체 폴백 금지 → 멘토 연결
            reason = verify_result.get("reason", "검증된 사용 가능 답변이 없습니다")
            print("  관찰3 | usable_answer_ids 비어있음 → 멘토 연결 (retrieved 전체 폴백 금지)")
            retrieval_log["verdict"]   = "mentor_needed"
            retrieval_log["avg_score"] = 0.0
            return self._mentor_fallback(
                session_id, strategy,
                fallback_type="no_usable_answers",
                reason=reason,
                mentor_match_hints={
                    "risk_flags":         risk_flags,
                    "question_units":     [u.get("unit", "") for u in question_units],
                    "current_bottleneck": current_bottleneck,
                },
                retrieval_log=retrieval_log,
            )

        usable_retrieved = [retrieved[i-1] for i in usable_ids]
        retrieval_log["usable_count"] = len(usable_retrieved)
        retrieval_log["stale_ids"]    = stale_ids

        # 가중 합산 (LLM 3개 + 코드 recency)
        avg_score = (
            safe_float(scores.get("relevance",           0.0)) * weights["relevance"]
            + safe_float(scores.get("evidence_sufficiency", 0.0)) * weights["evidence_sufficiency"]
            + safe_float(scores.get("situation_fit",        0.0)) * weights["situation_fit"]
            + recency_avg * weights["recency"]
        )
        privacy_safe = (not pre_privacy_unsafe) and safe_bool(scores.get("privacy_safe"), default=False)

        threshold     = self.THRESHOLD[strategy]     + threshold_boost
        mid_threshold = self.MID_THRESHOLD[strategy] + threshold_boost
        retrieval_log["avg_score"] = round(avg_score, 4)

        print(
            f"  행동2 | 가중합산: {avg_score:.2f} / "
            f"구간: [{mid_threshold:.2f}, {threshold:.2f}] / "
            f"usable: {len(usable_retrieved)}개 / "
            f"stale: {stale_ids or '없음'} / "
            f"개인정보 안전: {privacy_safe}"
        )
        if verify_result.get("privacy_risk_reason") and verify_result["privacy_risk_reason"] != "없음":
            print(f"         privacy 위험 이유: {verify_result['privacy_risk_reason']}")

        # ─────────────────────────────────────────
        # 관찰4 + 행동3: verdict 결정 → 답변 생성
        # ─────────────────────────────────────────
        stale_notice = _build_stale_notice(stale_ids)
        source_trace = {
            "used_answer_ids":      [a.get("answer_id") for a in usable_retrieved],
            "stale_but_useful_ids": stale_ids,
        }

        if avg_score >= threshold and privacy_safe:
            print(f"  관찰4 | 임계값 통과 ({avg_score:.2f} >= {threshold:.2f}) → 직접 답변")
            answer = self._generate_answer(
                refined_question, safe_context, usable_retrieved, "direct",
                stale_notice=stale_notice,
            )
            if answer is None:
                retrieval_log["verdict"] = "mentor_needed"
                return self._mentor_fallback(
                    session_id, strategy,
                    fallback_type="faithfulness_failed",
                    reason="답변 품질 기준 미달 (faithfulness 2회 실패)",
                    retrieval_log=retrieval_log,
                )
            retrieval_log["verdict"] = "llm_direct"
            update_session(session_id, {
                "answer_status":     "llm_direct",
                "llm_direct_answer": answer,
                "retrieval_log":     retrieval_log,
            })
            return {
                "verdict":            "llm_direct",
                "answer":             answer,
                "strategy":           strategy,
                "retrieved_count":    len(retrieved),
                "avg_score":          avg_score,
                "fallback_type":      None,
                "fallback_reason":    None,
                "mentor_match_hints": {},
                "source_trace":       source_trace,
                "retrieval_log":      retrieval_log,
            }

        elif avg_score >= mid_threshold and privacy_safe:
            print(f"  관찰4 | 중간 구간 ({mid_threshold:.2f} <= {avg_score:.2f} < {threshold:.2f}) → 부분 답변 + 멘토 권유")
            answer = self._generate_answer(
                refined_question, safe_context, usable_retrieved, "partial",
                stale_notice=stale_notice,
            )
            if answer is None:
                retrieval_log["verdict"] = "mentor_needed"
                return self._mentor_fallback(
                    session_id, strategy,
                    fallback_type="faithfulness_failed",
                    reason="답변 품질 기준 미달 (faithfulness 2회 실패)",
                    retrieval_log=retrieval_log,
                )
            retrieval_log["verdict"] = "partial_with_mentor_suggest"
            update_session(session_id, {
                "answer_status":      "partial_with_mentor_suggest",
                "llm_partial_answer": answer,
                "mentor_suggested":   True,
                "retrieval_log":      retrieval_log,
            })
            return {
                "verdict":            "partial_with_mentor_suggest",
                "answer":             answer,
                "strategy":           strategy,
                "retrieved_count":    len(retrieved),
                "avg_score":          avg_score,
                "fallback_type":      None,
                "fallback_reason":    None,
                "mentor_match_hints": {},
                "source_trace":       source_trace,
                "retrieval_log":      retrieval_log,
            }

        else:
            reason = verify_result.get("reason", "검색된 답변이 현재 질문에 충분히 맞지 않습니다")
            retrieval_log["verdict"] = "mentor_needed"
            return self._mentor_fallback(
                session_id, strategy,
                fallback_type="score_below_threshold",
                reason=reason,
                mentor_match_hints={
                    "avg_score":          round(avg_score, 4),
                    "risk_flags":         risk_flags,
                    "question_units":     [u.get("unit", "") for u in question_units],
                    "current_bottleneck": current_bottleneck,
                    "privacy_safe":       privacy_safe,
                },
                retrieval_log=retrieval_log,
            )

    # ── 내부 메서드 ──

    def _mentor_fallback(
        self,
        session_id: str,
        strategy: str,
        fallback_type: str = "unknown",
        reason: str = "",
        mentor_match_hints: dict | None = None,
        retrieval_log: dict | None = None,
    ) -> dict:
        """
        구조화된 멘토 연결 폴백.

        fallback_type:
          'requires_artifact_review' — 포트폴리오·이력서 검토 필요
          'no_similar_answers'       — DB에 유사 답변 없음
          'search_error'             — 임베딩 검색 실패
          'no_usable_answers'        — LLM이 usable_ids=[] 반환
          'score_below_threshold'    — 가중합산 점수 미달
          'faithfulness_failed'      — 답변 생성 품질 2회 미달

        mentor_match_hints: Agent 3 멘토 매칭에 활용할 정보
        """
        mentor_match_hints = mentor_match_hints or {}
        print(f"  → 멘토 연결 | type={fallback_type} / 사유: {reason}")
        update_session(session_id, {
            "answer_status":          "mentor_matched",
            "mentor_fallback_reason": reason,
            "fallback_type":          fallback_type,
            "mentor_match_hints":     mentor_match_hints,
            "retrieval_log":          retrieval_log or {},
        })
        return {
            "verdict":            "mentor_needed",
            "answer":             None,
            "strategy":           strategy,
            "retrieved_count":    retrieval_log.get("retrieved_count", 0) if retrieval_log else 0,
            "avg_score":          retrieval_log.get("avg_score", 0.0) if retrieval_log else 0.0,
            "fallback_type":      fallback_type,
            "fallback_reason":    reason,           # str (pipeline에서 직접 읽음)
            "mentor_match_hints": mentor_match_hints,
            "retrieval_log":      retrieval_log or {},
        }

    def _judge_strategy(self, question: str) -> dict:
        try:
            prompt   = COMPLEXITY_PROMPT.format(question=question)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"  _judge_strategy 실패: {e} → mentor_first 폴백")
            return {
                "strategy": "mentor_first", "complexity": "high",
                "personal_context_strength": "strong",
                "strategy_confidence": 0.0,
                "reason": "판단 실패",
            }

    def _verify(
        self,
        question: str,
        context: str,
        retrieved: list,
        current_bottleneck: str = "",
        expected_answer_type: str = "",
        question_units: list[dict] | None = None,
        risk_flags: list[str] | None = None,
    ) -> dict:
        try:
            answers_text = "\n\n".join(
                f"[답변 {i+1}] {a.get('answer_summarize', a.get('answer_content', ''))[:300]}"
                for i, a in enumerate(retrieved)
            )
            units_text = (
                "; ".join(u.get("unit", "") for u in (question_units or []))
                or "없음"
            )
            prompt = VERIFY_PROMPT.format(
                question=question,
                context=context,
                retrieved_answers=answers_text,
                current_bottleneck=current_bottleneck or "없음",
                expected_answer_type=expected_answer_type or "없음",
                question_units=units_text,
                risk_flags=", ".join(risk_flags) if risk_flags else "없음",
            )
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"  _verify 실패: {e} → 최저 점수 폴백")
            return {
                "scores": {
                    "relevance": 0.0, "evidence_sufficiency": 0.0,
                    "situation_fit": 0.0, "privacy_safe": False,
                },
                "usable_answer_ids":    [],
                "stale_but_useful_ids": [],
                "privacy_risk_reason":  "검증 실패",
                "reason":               "검증 실패",
            }

    def _generate_answer(
        self,
        question: str,
        context: str,
        retrieved: list,
        mode: str,
        stale_notice: str = "",
    ) -> str | None:
        answers_text = "\n\n".join(
            f"[답변 {i+1}] {a.get('answer_content', '')[:500]}"
            for i, a in enumerate(retrieved)
        )
        template = DIRECT_ANSWER_PROMPT if mode == "direct" else PARTIAL_ANSWER_PROMPT
        prompt   = template.format(
            question=question,
            context=context,
            retrieved_answers=answers_text,
            stale_notice=stale_notice,
        )
        grounded_threshold = self.GROUNDED_THRESHOLD[mode]

        generated = None
        for attempt in range(1, 3):
            try:
                response  = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3 if attempt == 2 else 0.5,
                )
                generated = response.choices[0].message.content
            except Exception as e:
                print(f"  답변 생성 실패 ({attempt}회차): {e}")
                continue

            # v2 신규: 생성된 답변에도 개인정보 정규식 검사
            if _has_privacy_risk(generated):
                print(f"  faithfulness | privacy 위험 감지 (생성 답변 정규식) → 재생성 ({attempt}회차)")
                generated = None
                continue

            faith = self._check_faithfulness(answers_text, generated, grounded_threshold)
            print(
                f"  faithfulness | grounded={faith.get('grounded', 0):.2f}"
                f" / supplement_marked={faith.get('supplement_marked')}"
                f" / factually_sound={faith.get('factually_sound')}"
                f" / over_generalized={faith.get('over_generalized')}"
                f" / privacy_leaked={faith.get('privacy_leaked')}"
                f" / pass={faith.get('pass')} ({attempt}회차)"
            )

            if faith.get("pass"):
                return generated

            print(f"  faithfulness 미통과 → {'재생성' if attempt < 2 else '멘토 연결 폴백'} ({faith.get('reason', '')})")

        return None  # 2회 모두 실패 → 호출부에서 멘토 연결

    def _check_faithfulness(self, answers_text: str, generated: str, grounded_threshold: float) -> dict:
        try:
            prompt   = FAITHFULNESS_PROMPT.format(
                retrieved_answers=answers_text,
                generated_answer=generated,
                grounded_threshold=grounded_threshold,
            )
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
            )
            result = json.loads(response.choices[0].message.content)

            # pass는 프롬프트 지시 + 코드에서 재계산 (이중 보정)
            code_pass = (
                safe_float(result.get("grounded", 0)) >= grounded_threshold
                and safe_bool(result.get("supplement_marked"), True)
                and safe_bool(result.get("factually_sound"), False)
                and not safe_bool(result.get("privacy_leaked"), True)
                and not safe_bool(result.get("over_generalized"), True)
            )
            result["pass"] = code_pass
            return result

        except Exception as e:
            print(f"  _check_faithfulness 실패: {e} → pass=False 폴백")
            return {
                "grounded": 0.0, "supplement_marked": False, "factually_sound": False,
                "privacy_leaked": False, "over_generalized": True,
                "pass": False, "reason": "검증 실패",
            }
