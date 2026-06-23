"""
멘티 통합 컨텍스트 빌더

가입 시 수집 프로필(mentees.json) + 세션 수집 정보(collected_info, routing_hints)를 병합해
Agent들이 공통으로 쓸 수 있는 통합 컨텍스트 딕셔너리를 생성한다.

우선순위:
  1. 세션 중 수집된 정보 (가장 최신·구체적)
  2. 가입 시 수집 프로필 (베이스라인 — 세션 정보가 비어있을 때 보완)

사용 예:
  from utils.context import build_refinement_context

  ctx = build_refinement_context(
      mentee_id=agent.mentee_id,
      collected_info=agent.collected_info,
      routing_hints=agent.routing_hints,
      safe_context=agent.safe_context,
  )
  # ctx["interest_domain"], ctx["target_role"], ctx["safe_context"] 등으로 접근
"""

from __future__ import annotations
from db.json_db import get_mentee


def build_refinement_context(
    mentee_id: str | None,
    collected_info: dict | None = None,
    routing_hints: dict | None = None,
    safe_context: str | None = None,
) -> dict:
    """
    통합 멘티 컨텍스트 생성

    Returns:
        {
            "interest_domain":          [...],
            "target_role":              "...",
            "career_stage":             "...",
            "desired_help":             "...",
            "transition_type":          "...",
            "constraints":              [...],
            "personal_context_strength": "weak/moderate/strong",
            "safe_context":             "...",   # 비식별 공유용 요약
            "source":                   "session_only" | "signup_only" | "merged",
        }
    """
    collected_info = collected_info or {}
    routing_hints  = routing_hints  or {}

    # ── 1. 세션 중 수집 정보 ──
    session_ctx: dict = {
        "interest_domain":           routing_hints.get("interest_domain", []),
        "target_role":               routing_hints.get("target_role", ""),
        "career_stage":              routing_hints.get("career_stage", "미상"),
        "desired_help":              routing_hints.get("desired_help", ""),
        "transition_type":           routing_hints.get("transition_type", "미상"),
        "constraints":               routing_hints.get("constraints", []),
        "personal_context_strength": routing_hints.get("personal_context_strength", "weak"),
        "safe_context":              safe_context or "",
    }

    # ── 2. 가입 프로필 병합 (있을 경우) ──
    signup_profile = get_mentee(mentee_id) if mentee_id else None

    if not signup_profile:
        session_ctx["source"] = "session_only"
        return session_ctx

    merged = dict(session_ctx)

    # 세션 값이 비어있을 때만 가입 프로필로 보완 (세션 정보가 항상 우선)
    if not merged["interest_domain"]:
        signup_domains = signup_profile.get("interest_domains", [])
        merged["interest_domain"] = signup_domains if isinstance(signup_domains, list) else []

    if not merged["target_role"]:
        merged["target_role"] = signup_profile.get("target_role", "")

    if merged["career_stage"] == "미상":
        merged["career_stage"] = signup_profile.get("career_stage", "미상")

    if not merged["desired_help"]:
        merged["desired_help"] = signup_profile.get("desired_help", "")

    if merged["transition_type"] == "미상":
        merged["transition_type"] = signup_profile.get("transition_type", "미상")

    if not merged["constraints"]:
        signup_constraints = signup_profile.get("constraints", [])
        merged["constraints"] = signup_constraints if isinstance(signup_constraints, list) else []

    if not merged["safe_context"]:
        # 가입 시 저장해둔 프로필 요약이 있으면 사용
        merged["safe_context"] = signup_profile.get("profile_summary", "")

    merged["source"] = "merged"
    return merged


def format_context_for_llm(ctx: dict) -> str:
    """
    build_refinement_context 결과를 LLM 프롬프트에 삽입할 텍스트로 변환
    safe_context 기반 — private_profile 포함하지 않음
    """
    parts = []

    if ctx.get("safe_context"):
        parts.append(ctx["safe_context"])
    else:
        # safe_context 없으면 routing_hints 필드들로 구성
        if ctx.get("target_role"):
            parts.append(f"관심 직무: {ctx['target_role']}")
        if ctx.get("interest_domain"):
            parts.append(f"관심 도메인: {', '.join(ctx['interest_domain'])}")
        if ctx.get("career_stage") and ctx["career_stage"] != "미상":
            parts.append(f"진로 단계: {ctx['career_stage']}")
        if ctx.get("desired_help"):
            parts.append(f"원하는 도움: {ctx['desired_help']}")
        if ctx.get("transition_type") not in ("없음", "미상", "", None):
            parts.append(f"전환 유형: {ctx['transition_type']}")
        if ctx.get("constraints"):
            parts.append(f"제약 조건: {', '.join(ctx['constraints'])}")

    return " / ".join(parts) if parts else "맥락 정보 없음"
