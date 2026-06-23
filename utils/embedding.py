"""
임베딩 생성 및 유사도 계산 유틸
- 모델: text-embedding-3-small (OpenAI)
- 유사도: numpy cosine similarity
"""

import os
import numpy as np
from openai import OpenAI

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def get_embedding(text: str) -> list[float]:
    """텍스트를 임베딩 벡터로 변환"""
    client = _get_client()
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text.strip(),
    )
    return response.data[0].embedding


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """두 벡터의 cosine similarity (-1 ~ 1)"""
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def top_k_similar(
    query_vec: list[float],
    candidates: list[dict],
    vec_field: str,
    k: int = 10,
) -> list[tuple[dict, float]]:
    """
    candidates 중 query_vec과 가장 유사한 상위 k개 반환
    각 candidate는 vec_field에 임베딩 리스트를 가지고 있어야 함
    반환: [(candidate_dict, score), ...] 내림차순
    """
    scored = []
    for c in candidates:
        vec = c.get(vec_field)
        if vec is None:
            continue
        score = cosine_similarity(query_vec, vec)
        scored.append((c, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


def build_mentor_text(mentor: dict, experiences: list[dict], answer_summaries: list[str]) -> str:
    """하위 호환용 — 단일 통합 텍스트 (레거시)"""
    return "\n".join(filter(None, [
        build_profile_text(mentor),
        build_career_text(experiences),
        build_answer_text(answer_summaries),
    ]))


def build_profile_text(mentor: dict) -> str:
    """채널 1: 멘토 프로필 텍스트 (직무·전문성·한줄 소개)"""
    parts = [
        mentor.get("matching_summary_text", ""),
        f"현재 직무: {mentor.get('current_role', '')}",
        f"경력: {mentor.get('years_of_experience', 0)}년",
    ]
    domain_tags = mentor.get("domain_tags", [])
    if domain_tags:
        parts.append(f"전문 분야: {', '.join(domain_tags)}")
    return "\n".join(p for p in parts if p.strip())


def build_career_text(experiences: list[dict]) -> str:
    """채널 2: 경력 경로 텍스트 (어떤 조직에서 무슨 역할을 했는지)"""
    if not experiences:
        return ""
    parts = []
    for exp in experiences:
        line = f"{exp.get('title', '')} ({exp.get('organization', '')}, {exp.get('period', '')})"
        desc = exp.get("description", "")
        if desc:
            line += f" - {desc}"
        parts.append(line)
    return "\n".join(parts)


def build_answer_text(answer_summaries: list[str]) -> str:
    """채널 3: 기존 답변 요약 텍스트 (어떤 질문에 어떤 답을 했는지)"""
    if not answer_summaries:
        return ""
    return "\n".join(f"[답변 경험] {s}" for s in answer_summaries)
