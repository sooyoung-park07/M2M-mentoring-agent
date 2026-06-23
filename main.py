"""
맨투맨(M2M) - AI 진로 멘토링 서비스 프로토타입
실행: python main.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from agents.question_refine_agent import QuestionRefineAgent
from agents.search_verify_agent import SearchVerifyAgent
from agents.mentor_match_agent import MentorMatchAgent
from agents.assetize import AssetizeAgent, update_satisfaction
from db.json_db import (
    get_mentee, update_mentee_persistent_bottleneck,
    update_mentee_transition_profile, increment_mentee_session_count,
    create_question_session, update_session, get_session, new_id,
)


# ─────────────────────────────────────────
# Pipeline context builders
# ─────────────────────────────────────────

def build_pipeline_context(agent1, mentee_profile: dict | None = None) -> dict:
    """
    Agent 1 종료 직후 호출.
    agent1 속성을 단일 dict로 통합 → Agent 2/3/4에 전달하는 공통 컨텍스트.
    mentee_profile이 있으면 장기 프로필의 transition_profile을 fallback으로 활용.
    """
    routing_hints = getattr(agent1, "routing_hints", {}) or {}
    taxonomy_tags = getattr(agent1, "taxonomy_tags", {}) or {}

    risk_flags = getattr(agent1, "risk_flags", routing_hints.get("risk_flags", [])) or []
    if isinstance(risk_flags, str):
        risk_flags = [risk_flags]
    question_units = getattr(agent1, "question_units", []) or []

    # mentee_profile에서 transition_profile fallback
    mp_transition = {}
    if mentee_profile:
        mp_transition = mentee_profile.get("transition_profile", {}) or {}

    def _get(attr, hint_key, profile_key=None, default=""):
        val = getattr(agent1, attr, None)
        if not val:
            val = routing_hints.get(hint_key, None)
        if not val and profile_key and mp_transition:
            val = mp_transition.get(profile_key, None)
        return val or default

    hard_case_flags = {
        "requires_artifact_review": getattr(
            agent1, "requires_artifact_review",
            routing_hints.get("requires_artifact_review", False)
        ),
        "risk_flags":           risk_flags,
        "recency_sensitive":    getattr(agent1, "recency_sensitive",    routing_hints.get("recency_sensitive", False)),
        "scope_too_broad":      getattr(agent1, "scope_too_broad",      routing_hints.get("scope_too_broad", False)),
        "source_role":          _get("source_role",         "source_role",         "source_role"),
        "target_role":          routing_hints.get("target_role",         mp_transition.get("target_role", "")),
        "target_role_specificity": routing_hints.get("target_role_specificity", "unclear"),
        "bridge_hypothesis":    _get("bridge_hypothesis",   "bridge_hypothesis",   "bridge_hypothesis"),
        "transferable_skills":  (
            getattr(agent1, "transferable_skills", None)
            or routing_hints.get("transferable_skills", None)
            or mp_transition.get("transferable_skills", [])
        ) or [],
        "target_domain_candidates": (
            getattr(agent1, "target_domain_candidates", None)
            or routing_hints.get("target_domain_candidates", None)
            or mp_transition.get("target_domain_candidates", [])
        ) or [],
        "question_structure":  routing_hints.get("question_structure",  ""),
        "document_help_type":  routing_hints.get("document_help_type",  ""),
        "recency_level":       routing_hints.get("recency_level",       ""),
        "recency_reason":      routing_hints.get("recency_reason",      ""),
    }

    return {
        "session_id":           agent1.session_id,
        "mentee_id":            getattr(agent1, "mentee_id", ""),
        "refined_question":     getattr(agent1, "refined_question",     ""),
        "conversation_summary": getattr(agent1, "conversation_summary", ""),
        "safe_context":         getattr(agent1, "safe_context", None)
                                or getattr(agent1, "conversation_summary", ""),
        "search_query":         getattr(agent1, "search_query", None),
        "match_query":          getattr(agent1, "match_query",  None),
        "routing_hints":        routing_hints,
        "taxonomy_tags":        taxonomy_tags,
        "current_bottleneck":   getattr(agent1, "current_bottleneck",   ""),
        "expected_answer_type": getattr(agent1, "expected_answer_type", ""),
        "question_units":       question_units,
        "hard_case_flags":      hard_case_flags,
        "source_role":          hard_case_flags["source_role"],
        "target_role":          hard_case_flags["target_role"],
        "bridge_hypothesis":    hard_case_flags["bridge_hypothesis"],
        "transferable_skills":  hard_case_flags["transferable_skills"],
        "target_domain_candidates": hard_case_flags["target_domain_candidates"],
        "risk_flags":           risk_flags,
    }


def save_question_session_from_agent1(agent1, mentee_id: str) -> dict:
    """
    Agent 1 완료 후 question_sessions.json에 세션 저장.
    반환값의 session_id를 이후 파이프라인 전체에서 사용한다.
    """
    routing_hints = getattr(agent1, "routing_hints", {}) or {}
    risk_flags    = getattr(agent1, "risk_flags", routing_hints.get("risk_flags", [])) or []
    if isinstance(risk_flags, str):
        risk_flags = [risk_flags]

    hard_case_flags = {
        "requires_artifact_review": getattr(agent1, "requires_artifact_review",
                                            routing_hints.get("requires_artifact_review", False)),
        "recency_sensitive":        getattr(agent1, "recency_sensitive",
                                            routing_hints.get("recency_sensitive", False)),
        "scope_too_broad":          getattr(agent1, "scope_too_broad",
                                            routing_hints.get("scope_too_broad", False)),
        "risk_flags":               risk_flags,
        "question_structure":       routing_hints.get("question_structure", ""),
        "document_help_type":       routing_hints.get("document_help_type", ""),
        "recency_level":            routing_hints.get("recency_level", ""),
        "recency_reason":           routing_hints.get("recency_reason", ""),
        "source_role":              getattr(agent1, "source_role",         routing_hints.get("source_role", "")),
        "target_role":              routing_hints.get("target_role", ""),
        "target_role_specificity":  routing_hints.get("target_role_specificity", "unclear"),
        "bridge_hypothesis":        getattr(agent1, "bridge_hypothesis",   routing_hints.get("bridge_hypothesis", "")),
        "transferable_skills":      getattr(agent1, "transferable_skills", routing_hints.get("transferable_skills", [])) or [],
        "target_domain_candidates": getattr(agent1, "target_domain_candidates", routing_hints.get("target_domain_candidates", [])) or [],
    }

    session = create_question_session(
        mentee_id=mentee_id,
        refined_question=     getattr(agent1, "refined_question",     ""),
        conversation_summary= getattr(agent1, "conversation_summary", ""),
        safe_context=         getattr(agent1, "safe_context", None) or getattr(agent1, "conversation_summary", ""),
        search_query=         getattr(agent1, "search_query", None),
        match_query=          getattr(agent1, "match_query",  None),
        current_bottleneck=   getattr(agent1, "current_bottleneck",   ""),
        expected_answer_type= getattr(agent1, "expected_answer_type", ""),
        question_units=       getattr(agent1, "question_units",       []) or [],
        taxonomy_tags=        getattr(agent1, "taxonomy_tags",        {}) or {},
        routing_hints=        routing_hints,
        hard_case_flags=      hard_case_flags,
    )
    # agent1.session_id를 새로 생성된 session_id로 덮어씀
    agent1.session_id = session["session_id"]
    return session


def update_mentee_profile_from_agent1(mentee_id: str, agent1) -> None:
    """
    Agent 1 완료 후 멘티 장기 프로필(mentees.json)을 업데이트.
    세션마다 달라지는 값은 저장하지 않고, 장기적으로 누적할 값만 반영.
    """
    routing_hints = getattr(agent1, "routing_hints", {}) or {}

    # persistent_bottlenecks 누적
    bottleneck = getattr(agent1, "current_bottleneck", "")
    if bottleneck:
        update_mentee_persistent_bottleneck(mentee_id, bottleneck)

    # transition_profile 갱신 (비어있지 않은 값만)
    transition_updates = {}
    for field, getter in [
        ("source_role",              lambda: getattr(agent1, "source_role",         routing_hints.get("source_role", ""))),
        ("target_role",              lambda: routing_hints.get("target_role", "")),
        ("bridge_hypothesis",        lambda: getattr(agent1, "bridge_hypothesis",   routing_hints.get("bridge_hypothesis", ""))),
        ("transferable_skills",      lambda: getattr(agent1, "transferable_skills", routing_hints.get("transferable_skills", [])) or []),
        ("target_domain_candidates", lambda: getattr(agent1, "target_domain_candidates", routing_hints.get("target_domain_candidates", [])) or []),
    ]:
        val = getter()
        if val:
            transition_updates[field] = val
    if transition_updates:
        update_mentee_transition_profile(mentee_id, transition_updates)

    # 세션 카운트 증가
    increment_mentee_session_count(mentee_id)


def build_mentee_constraints(ctx: dict, search_result: dict | None = None,
                              mentee_profile: dict | None = None) -> dict:
    """
    Agent 3 호출용 mentee_constraints 생성.
    ctx(session) + search_result(Agent 2) + mentee_profile(장기) 3-layer 병합.
    우선순위: session ctx > search_result > mentee_profile
    """
    routing_hints   = ctx.get("routing_hints",   {})
    hard_case_flags = ctx.get("hard_case_flags", {})
    taxonomy_tags   = ctx.get("taxonomy_tags",   {})
    search_result   = search_result or {}
    mentor_match_hints = search_result.get("mentor_match_hints", {}) or {}

    mp_transition  = {}
    mp_interest    = {}
    if mentee_profile:
        mp_transition = mentee_profile.get("transition_profile",  {}) or {}
        mp_interest   = mentee_profile.get("interest_profile",    {}) or {}

    def _first(*vals, default=""):
        for v in vals:
            if v:
                return v
        return default

    return {
        "match_query":          ctx.get("match_query") or ctx.get("refined_question"),
        "interest_domain":      (routing_hints.get("interest_domain")
                                 or taxonomy_tags.get("domain_tags")
                                 or mp_interest.get("interest_domain", [])),
        "target_role":          _first(routing_hints.get("target_role"),
                                       hard_case_flags.get("target_role"),
                                       mp_transition.get("target_role")),
        "target_role_specificity": _first(hard_case_flags.get("target_role_specificity"),
                                          routing_hints.get("target_role_specificity"),
                                          default="unclear"),
        "source_role":              _first(hard_case_flags.get("source_role"),
                                           mp_transition.get("source_role")),
        "bridge_hypothesis":        _first(hard_case_flags.get("bridge_hypothesis"),
                                           mp_transition.get("bridge_hypothesis")),
        "transferable_skills":      (hard_case_flags.get("transferable_skills")
                                     or mp_transition.get("transferable_skills", [])),
        "target_domain_candidates": (hard_case_flags.get("target_domain_candidates")
                                     or mp_transition.get("target_domain_candidates", [])),
        "transition_type":     routing_hints.get("transition_type", "미상"),
        "desired_help":        routing_hints.get("desired_help",    ""),
        "constraints":         routing_hints.get("constraints",     []),
        "current_bottleneck":  ctx.get("current_bottleneck",        ""),
        "expected_answer_type":ctx.get("expected_answer_type",       ""),
        "question_units":      ctx.get("question_units",             []),
        "risk_flags":          ctx.get("risk_flags",                 []),
        "taxonomy_tags":       taxonomy_tags,
        "fallback_type":       search_result.get("fallback_type",    ""),
        "fallback_reason":     search_result.get("fallback_reason",  ""),
        "mentor_match_hints":  mentor_match_hints,
        "needed_expertise":    mentor_match_hints.get("needed_expertise", []),
        # 멘티 장기 누적 병목 (Agent 3에서 참고용)
        "persistent_bottlenecks": (mentee_profile or {}).get("persistent_bottlenecks", []),
    }


# ─────────────────────────────────────────
# UX 유틸
# ─────────────────────────────────────────

def print_divider(title: str = "") -> None:
    print(f"\n{chr(9472) * 50}")
    if title:
        print(f"  {title}")
        print(f"{chr(9472) * 50}")


# ─────────────────────────────────────────
# 멘티 플로우
# ─────────────────────────────────────────

def run_mentee_flow(mentee_id: str | None = None) -> None:
    if mentee_id is None:
        mentee_id = new_id("mt_")

    # 멘티 장기 프로필 로드 (없으면 None)
    mentee_profile = None
    from db.json_db import get_mentee as _get_mentee
    mentee_profile = _get_mentee(mentee_id)

    print_divider("STEP 1 | 질문 정제 에이전트")

    agent = QuestionRefineAgent(mentee_id=mentee_id)
    intro = "안녕! 나는 맸투맸 진로 상담 에이전트야. 어떤 진로 고민이 있는지 편하게 얘기해줘"
    print(f"\n에이전트: {intro}\n")

    while True:
        user_input = input("나: ").strip()
        if user_input.lower() in ("q", "quit", "exit"):
            print("종료합니다.")
            return
        if not user_input:
            continue
        response = agent.chat(user_input)
        print(f"\n에이전트: {response}\n")
        if getattr(agent, "is_done", False):
            break

    # ── Agent 1 완료: 세션 저장 + 멘티 프로필 갱신 ──
    session = save_question_session_from_agent1(agent, mentee_id)
    update_mentee_profile_from_agent1(mentee_id, agent)

    # ── Pipeline context 구성 ──
    ctx = build_pipeline_context(agent, mentee_profile=mentee_profile)
    routing_hints = ctx["routing_hints"]

    if routing_hints.get("interest_domain"):
        print(f"  [멘티 관심 직무] {routing_hints['interest_domain']}")
    if ctx["target_role"]:
        print(f"  [목표 직무] {ctx['target_role']}")
    if routing_hints.get("transition_type") not in ("없음", "미상", "", None):
        print(f"  [전환 유형] {routing_hints['transition_type']}")

    print_divider("STEP 2 | 검색·검증 에이전트")

    sv_agent = SearchVerifyAgent()
    result = sv_agent.run(
        session_id=           ctx["session_id"],
        refined_question=     ctx["refined_question"],
        conversation_summary= ctx["conversation_summary"],
        routing_hints=        ctx["routing_hints"],
        search_query=         ctx["search_query"],
        safe_context=         ctx["safe_context"],
        current_bottleneck=   ctx["current_bottleneck"],
        expected_answer_type= ctx["expected_answer_type"],
        question_units=       ctx["question_units"],
        hard_case_flags=      ctx["hard_case_flags"],
    )

    need_mentor = False

    if result["verdict"] == "llm_direct":
        print_divider("결과 | AI 참고 답변")
        print(result["answer"])
        print()
        feedback = input("답변이 도움이 됐나요? (1~5, 건너뛰려면 엔터): ").strip()
        if feedback.isdigit() and 1 <= int(feedback) <= 5:
            score = (int(feedback) - 1) / 4
            update_session(ctx["session_id"], {"llm_feedback_score": score})
            print("피드백 저장 완료!")

    elif result["verdict"] == "partial_with_mentor_suggest":
        print_divider("결과 | AI 부분 답변 (개인 맥락 한계 있음)")
        print(result["answer"])
        print()
        feedback = input("답변이 도움이 됐나요? (1~5, 건너뛰려면 엔터): ").strip()
        if feedback.isdigit() and 1 <= int(feedback) <= 5:
            score = (int(feedback) - 1) / 4
            update_session(ctx["session_id"], {"answer_status": "partial_answer", "llm_feedback_score": score})
            print("피드백 저장 완료!")
        go_mentor = input("\n현직자 멘토에게 직접 연결할까요? (y/엔터): ").strip().lower()
        need_mentor = (go_mentor == "y")

    else:
        need_mentor = True

    if need_mentor:
        print_divider("STEP 3 | 멘토 매칭 에이전트")
        mentee_constraints = build_mentee_constraints(ctx, result, mentee_profile=mentee_profile)
        mm_agent = MentorMatchAgent()
        match_result = mm_agent.run(
            session_id=           ctx["session_id"],
            refined_question=     ctx["refined_question"],
            conversation_summary= ctx["safe_context"],
            mentee_constraints=   mentee_constraints,
        )

        if not match_result["top3"]:
            print("현재 적합한 멘토를 찾지 못했어. 나중에 다시 시도해줘.")
            return

        print_divider("결과 | 추천 멘토 Top-3")
        for item in match_result["top3"]:
            name   = item.get("mentor_info", {}).get("name", "알 수 없음")
            role   = item.get("current_role", "")
            reason = item.get("recommendation_reason", "")
            print(f"\n[{item['rank']}순위] {name} | {role}")
            print(f"추천 이유: {reason}")

        print()
        choice = input("연결할 멘토 번호를 입력해줘 (1~3, 건너뛰려면 엔터): ").strip()
        if choice in ("1", "2", "3"):
            idx = int(choice) - 1
            selected = match_result["top3"][idx]
            print(f"\n✓ {selected.get('mentor_info', {}).get('name', '')} 멘토에게 정제된 질문을 전달했어!")
            print(f"질문: {ctx['refined_question']}")

        print_divider("STEP 4 | 멘토 답변 자산화 (시뮬레이션)")
        print("멘토 답변을 입력해줘 (실제 서비스에서는 멘토 앱에서 입력됨)")
        print("(건너뛰려면 엔터)\n")
        answer_content = input("멘토 답변: ").strip()
        if answer_content:
            answer_summary    = input("답변 요약 (한 줄): ").strip() or answer_content[:100]
            domain_tags_input = input("도메인 태그 (쉼표 구분, 예: 데이터분석,비전공전환): ").strip()
            domain_tags = [t.strip() for t in domain_tags_input.split(",") if t.strip()]

            print("\n이 답변이 유사한 고민을 가진 다른 학생들에게 익명으로 참고자료로 활용될 수 있습니다.")
            mentor_consent = (input("멘토 동의 여부 (y/n): ").strip().lower() == "y")
            print("이 상담 내용이 다른 학생들을 위해 익명으로 활용될 수 있습니다.")
            mentee_consent = (input("멘티 동의 여부 (y/n): ").strip().lower() == "y")

            mentor_id = (
                match_result["top3"][0]["mentor_id"]
                if choice not in ("1", "2", "3")
                else match_result["top3"][int(choice) - 1]["mentor_id"]
            )

            result_record = AssetizeAgent().run(
                session_id=           ctx["session_id"],
                mentor_id=            mentor_id,
                question_content=     ctx["refined_question"],
                answer_content=       answer_content,
                answer_summarize=     answer_summary,
                domain_tags=          domain_tags,
                mentor_consent=       mentor_consent,
                mentee_consent=       mentee_consent,
                satisfaction_score=   None,
                taxonomy_tags=        ctx["taxonomy_tags"],
                source_role=          ctx["source_role"],
                target_role=          ctx["target_role"],
                current_bottleneck=   ctx["current_bottleneck"],
                expected_answer_type= ctx["expected_answer_type"],
                question_units=       ctx["question_units"],
            )
            if result_record.get("is_assetized"):
                print("✓ 답변이 자산 DB에 저장되어 다음 멘티에게 활용될 거야!")
            else:
                print(f"  답변은 저장됐지만 자산화 제외됐어. (사유: {result_record.get('reject_reason', '알 수 없음')})")


def run_setup() -> None:
    try:
        import openai
        import numpy
    except ImportError:
        print("필요한 패키지가 없어. 먼저 설치해줘:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY가 설정되지 않았어.")
        print(".env 파일에 OPENAI_API_KEY=sk-... 를 추가해줘")
        sys.exit(1)


def main() -> None:
    run_setup()
    print("=" * 50)
    print("  맸투맸(M2M) - AI 진로 멘토링 서비스")
    print("=" * 50)
    print("\n메뉴:")
    print("  1. 진로 상담 시작 (멘티 플로우)")
    print("  2. 페르소나 생성 (data/generate_personas.py 실행)")
    print("  q. 종료")
    choice = input("\n선택: ").strip()
    if choice == "1":
        run_mentee_flow()
    elif choice == "2":
        os.system(f"{sys.executable} {ROOT / 'data' / 'generate_personas.py'}")
    elif choice in ("q", "quit"):
        print("종료합니다.")
    else:
        print("잘못된 입력이야.")


if __name__ == "__main__":
    main()
