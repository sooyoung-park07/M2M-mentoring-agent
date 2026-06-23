"""
GPT-4o로 멘티 50명 + 멘토 50명 페르소나 자동 생성 (v3 — DB 2층 분리 스키마 반영)

실행:
  python data/generate_personas.py              # 누적 추가
  python data/generate_personas.py --replace    # 기존 records 삭제 후 재생성
  python data/generate_personas.py --only mentee
  python data/generate_personas.py --only mentor
"""

import os, json, sys, argparse
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from db.json_db import DB_DIR, new_id, now_str

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# ─────────────────────────────────────────────────────────────────
# 멘티 시드 (50명) — 실제 잇다 50건 질문 패턴 기반
# ─────────────────────────────────────────────────────────────────

MENTEE_SPECS = [
    # ── 금융/투자 (7명) ──────────────────────────────
    {"domain": "금융/투자", "question_type": "복합질문형+경험부족형", "career_stage": "취준생", "age": 24,
     "background_hint": "상경계열 졸업예정, CFA Level I 합격·학회 리서치 경험 있으나 유관 인턴 無. 자산운용사 지원 중 서류 반복 탈락.",
     "constraint_hint": "인턴 경험 부족, 서류 탈락 반복"},
    {"domain": "금융/투자", "question_type": "가능성판단형", "career_stage": "취준생", "age": 23,
     "background_hint": "비상경계열(공학) 전공이지만 금융권 이직 희망. 투자자산운용사 자격증 취득, 관련 대외활동 1건.",
     "constraint_hint": "비전공 전환, 정보 부족"},
    {"domain": "금융/투자", "question_type": "커리어로드맵형", "career_stage": "대학생", "age": 22,
     "background_hint": "경제학과 2학년, 증권사 리서치 또는 IB 목표. 어디서 무엇부터 준비해야 하는지 로드맵 요청.",
     "constraint_hint": "정보 부족, 진입 경로 불명확"},
    {"domain": "금융/투자", "question_type": "자료피드백형+경험부족형", "career_stage": "취준생", "age": 25,
     "background_hint": "상경계열 졸업, 증권사 운용 보조 지원. 자소서 작성 방법을 모르겠음. 인턴 경험 없음.",
     "constraint_hint": "자소서 전략 부재, 인턴 없음"},
    {"domain": "금융/투자", "question_type": "최신전망형", "career_stage": "취준생", "age": 24,
     "background_hint": "퀀트·데이터 기반 투자에 관심, AI가 애널리스트 직무를 대체할지 불안. 방향성 고민.",
     "constraint_hint": "직무 전망 불확실성"},
    {"domain": "금융/투자", "question_type": "직무미분화형", "career_stage": "취준생", "age": 23,
     "background_hint": "금융권 진출 희망인데 IB, 운용, 리서치, 핀테크 중 어디로 가야 할지 모르겠음.",
     "constraint_hint": "직무 미분화, 정보 과부하"},
    {"domain": "금융/투자", "question_type": "가능성판단형+직무전환형", "career_stage": "직장인", "age": 28,
     "background_hint": "일반 기업 재무팀 3년차. 증권사 또는 자산운용사로 이직을 원함. 실무 경험을 어떻게 어필할지 막막.",
     "constraint_hint": "경력 재해석 필요, 직무 전환 논리 부족"},
    # ── IT개발/데이터 (10명) ─────────────────────────
    {"domain": "IT개발/백엔드", "question_type": "최신전망형+복합질문형", "career_stage": "대학생", "age": 21,
     "background_hint": "컴퓨터공학과 3학년, 생성형 AI가 코딩 대체하는 시대에 백엔드 개발자로 살아남을 수 있을지 불안.",
     "constraint_hint": "직무 전망 불확실, 준비 우선순위 모호"},
    {"domain": "IT개발/백엔드", "question_type": "커리어로드맵형", "career_stage": "취준생", "age": 24,
     "background_hint": "비전공(경영학) 출신 백엔드 개발자 전환 희망. 부트캠프 수료, 포트폴리오 준비 중. 대기업 입사가 목표.",
     "constraint_hint": "비전공 전환, 포트폴리오 전략 부재"},
    {"domain": "데이터분석/AI", "question_type": "가능성판단형+경험부족형", "career_stage": "취준생", "age": 25,
     "background_hint": "인문계열(사회학) 전공, 파이썬·SQL 독학. 데이터 애널리스트 지원 가능한지 자신 없음.",
     "constraint_hint": "비전공, 실무 경험 부족"},
    {"domain": "데이터분석/AI", "question_type": "직무미분화형", "career_stage": "취준생", "age": 24,
     "background_hint": "통계학과 졸업예정. DA, DS, MLE 중 어디에 맞는지 모르겠음. 다 해보고 싶지만 방향이 없음.",
     "constraint_hint": "직무 미분화, 목표 불명확"},
    {"domain": "데이터분석/AI", "question_type": "자료피드백형+경험부족형", "career_stage": "취준생", "age": 23,
     "background_hint": "통계학·컴공 복수전공. 캐글 경험 있으나 실무 인턴 없음. 포트폴리오가 경쟁력 있는지 확신 없음.",
     "constraint_hint": "포트폴리오 수준 불확실"},
    {"domain": "IT개발/프론트엔드", "question_type": "커리어로드맵형", "career_stage": "대학생", "age": 22,
     "background_hint": "컴공 2학년, 스타트업 인턴 경험 1회. 개발자로 커리어를 쌓는 단계별 로드맵이 궁금.",
     "constraint_hint": "경험 부족, 단계 불명확"},
    {"domain": "데이터분석/AI", "question_type": "직무전환형+가능성판단형", "career_stage": "직장인", "age": 30,
     "background_hint": "마케터 4년차. SQL·파이썬 독학 후 데이터 분석가로 전환 가능한지 고민. 어떤 역량을 어필해야 할지 모름.",
     "constraint_hint": "직무 전환 논리 부족, 경력 재해석 필요"},
    {"domain": "IT개발/백엔드", "question_type": "복합질문형", "career_stage": "취준생", "age": 25,
     "background_hint": "컴공 졸업, 대기업 공채 준비 중. CS 지식, 코딩테스트, 프로젝트 경험 모두 부족한 것 같아 막막.",
     "constraint_hint": "준비 우선순위 불명확"},
    {"domain": "데이터분석/AI", "question_type": "최신전망형", "career_stage": "직장인", "age": 32,
     "background_hint": "BI 분석 3년차. ChatGPT 등 생성형 AI 도입으로 내 직무가 대체될지 걱정. 어떻게 포지셔닝해야 할지.",
     "constraint_hint": "직무 가치 하락 불안"},
    {"domain": "IT개발/백엔드", "question_type": "가능성판단형", "career_stage": "취준생", "age": 23,
     "background_hint": "지방대 컴공, 학점 낮음. 개발 실력은 있다고 생각하는데 대기업 서류조차 통과가 어려움.",
     "constraint_hint": "학벌·학점 핸디캡, 지방대"},
    # ── 마케팅/MD (10명) ─────────────────────────────
    {"domain": "브랜드마케팅", "question_type": "커리어로드맵형", "career_stage": "취준생", "age": 23,
     "background_hint": "스포츠 브랜드 마케터 목표. 관련 자격증보다 실제 경험이 중요하다는 걸 알지만 어디서 시작해야 하는지 모름.",
     "constraint_hint": "정보 부족, 진입 경로 불명확"},
    {"domain": "브랜드마케팅", "question_type": "경험부족형+자료피드백형", "career_stage": "취준생", "age": 24,
     "background_hint": "뷰티 마케터 지망. 직무 관련 인턴·대외활동 전무. 일반 경험(학과 활동, 아르바이트)만 있음.",
     "constraint_hint": "경험 공백, 자소서 소재 부재"},
    {"domain": "퍼포먼스마케팅", "question_type": "직무미분화형", "career_stage": "취준생", "age": 25,
     "background_hint": "브랜드마케팅·퍼포먼스마케팅·콘텐츠마케팅 중 어디로 가야 할지 모름. 여러 대외활동을 했지만 확신 없음.",
     "constraint_hint": "직무 미분화, 다방면 경험만 있음"},
    {"domain": "MD/유통마케팅", "question_type": "가능성판단형", "career_stage": "취준생", "age": 24,
     "background_hint": "패션 MD 지망, 관련 전공 아님(영문학). 제품 기획이나 바잉 업무를 해볼 수 있을지 확신 없음.",
     "constraint_hint": "비전공, 유관 경험 없음"},
    {"domain": "브랜드마케팅", "question_type": "복합질문형", "career_stage": "취준생", "age": 26,
     "background_hint": "대기업 브랜드 마케팅 지원 중. 자소서 스토리 구성부터 면접까지 통합적으로 어떻게 준비해야 할지 막막.",
     "constraint_hint": "전략 부재, 복합 준비 필요"},
    {"domain": "퍼포먼스마케팅", "question_type": "직무전환형", "career_stage": "직장인", "age": 29,
     "background_hint": "대행사 콘텐츠 마케터 3년차. 퍼포먼스 마케팅으로 이직하고 싶은데 GA4, Meta Ads 경험이 없음.",
     "constraint_hint": "스킬 갭, 이직 경쟁력 부족"},
    {"domain": "MD/유통마케팅", "question_type": "커리어로드맵형", "career_stage": "취준생", "age": 23,
     "background_hint": "이커머스 MD 목표. MD는 어떤 경험이 필요한지, 어디서부터 준비해야 하는지 정보가 없음.",
     "constraint_hint": "직무 이해 부족, 정보 부족"},
    {"domain": "브랜드마케팅", "question_type": "최신전망형", "career_stage": "취준생", "age": 24,
     "background_hint": "AI 콘텐츠 생성 도구가 마케터를 대체할 것인지 걱정. 마케터만의 차별점이 무엇인지 묻고 싶음.",
     "constraint_hint": "직무 가치 불안, 방향성 모색"},
    {"domain": "MD/유통마케팅", "question_type": "가능성판단형+경험부족형", "career_stage": "취준생", "age": 25,
     "background_hint": "유통·MD 지망이지만 관련 인턴 0. 아르바이트 경험만 있는데 취업 가능성이 있는지 모르겠음.",
     "constraint_hint": "경험 부족, 가능성 불확실"},
    {"domain": "퍼포먼스마케팅", "question_type": "자료피드백형", "career_stage": "취준생", "age": 24,
     "background_hint": "퍼포먼스 마케팅 포트폴리오 준비 중. 개인 프로젝트로 Meta Ads를 직접 운영해봤는데 어필 포인트가 맞는지 확인이 필요.",
     "constraint_hint": "포트폴리오 완성도 불확실"},
    # ── 전략기획/컨설팅 (5명) ─────────────────────────
    {"domain": "전략기획", "question_type": "직무미분화형", "career_stage": "취준생", "age": 25,
     "background_hint": "대기업 전략기획 vs 컨설팅 vs 스타트업 기획 중 고민. 어디로 가야 자신에게 맞는지 모르겠음.",
     "constraint_hint": "방향 미결, 비교 정보 부족"},
    {"domain": "전략기획", "question_type": "커리어로드맵형", "career_stage": "대학생", "age": 23,
     "background_hint": "경영학과 3학년. 전략기획자가 되고 싶은데 무엇을 준비해야 하는지 막막.",
     "constraint_hint": "정보 부족, 준비 경험 없음"},
    {"domain": "컨설팅", "question_type": "가능성판단형+경험부족형", "career_stage": "취준생", "age": 26,
     "background_hint": "이공계(산업공학) 출신으로 전략 컨설팅 입사 희망. 케이스 인터뷰 경험 없음.",
     "constraint_hint": "케이스 인터뷰 준비 부족"},
    {"domain": "전략기획", "question_type": "직무전환형", "career_stage": "직장인", "age": 31,
     "background_hint": "영업팀 5년차. 전략기획 부서로 사내 이동을 원하는데 어떤 역량을 어필해야 하는지 모름.",
     "constraint_hint": "경력 재포지셔닝 필요"},
    {"domain": "컨설팅", "question_type": "복합질문형", "career_stage": "취준생", "age": 25,
     "background_hint": "4대 컨설팅 MBB 지원 희망. 케이스 인터뷰, 영어, 글로벌 경험 모두 준비해야 하는데 무엇부터 해야 할지 모름.",
     "constraint_hint": "우선순위 불명확, 준비 요소 다수"},
    # ── HR/경영지원 (5명) ─────────────────────────────
    {"domain": "HR/인사", "question_type": "직무미분화형", "career_stage": "취준생", "age": 24,
     "background_hint": "HR 직무 중 채용, 교육, 노무, HRD 중 어디가 자신에게 맞는지 모름. 다양하게 해보고 싶지만 방향이 없음.",
     "constraint_hint": "직무 미분화"},
    {"domain": "HR/인사", "question_type": "경험부족형+가능성판단형", "career_stage": "취준생", "age": 23,
     "background_hint": "심리학과 출신으로 HR 직무 지원 희망. 유관 경험 없음. 전공이 어필될 수 있는지 모르겠음.",
     "constraint_hint": "유관 경험 없음, 전공 어필 불확실"},
    {"domain": "HR/인사", "question_type": "커리어로드맵형", "career_stage": "대학생", "age": 22,
     "background_hint": "경영학과 2학년. 사람이 좋아서 HR을 생각하는데 어떻게 커리어를 쌓아야 하는지 모름.",
     "constraint_hint": "정보 부족, 직무 이해 초기"},
    {"domain": "HR/인사", "question_type": "직무전환형", "career_stage": "직장인", "age": 33,
     "background_hint": "대기업 총무팀 6년차. HR 업무로 전환 희망. 직무 전환에 필요한 스킬과 전략을 모름.",
     "constraint_hint": "직무 전환 장벽"},
    {"domain": "HR/인사", "question_type": "최신전망형", "career_stage": "취준생", "age": 25,
     "background_hint": "HR 직무가 AI 자동화로 대체될지 걱정. People Analytics나 HRTech로 전환해야 하는지 고민.",
     "constraint_hint": "직무 전망 불안"},
    # ── 연구/개발/엔지니어링 (5명) ───────────────────
    {"domain": "연구개발/R&D", "question_type": "커리어로드맵형", "career_stage": "대학원생", "age": 26,
     "background_hint": "석사 졸업 예정. 기업 R&D와 연구소 중 어디로 가야 할지 결정이 안 됨.",
     "constraint_hint": "방향 선택 어려움"},
    {"domain": "생산/품질/제조", "question_type": "가능성판단형+경험부족형", "career_stage": "취준생", "age": 24,
     "background_hint": "기계공학과 졸업예정. 대기업 생산관리 지원 중. 현장 실습 경험 없음.",
     "constraint_hint": "실무 경험 없음"},
    {"domain": "연구개발/R&D", "question_type": "직무미분화형", "career_stage": "대학원생", "age": 27,
     "background_hint": "화학공학 석사. 소재/반도체/제약 R&D 중 어디로 가야 할지 고민.",
     "constraint_hint": "산업 선택 미결"},
    {"domain": "생산/품질/제조", "question_type": "직무전환형", "career_stage": "직장인", "age": 34,
     "background_hint": "중견기업 생산팀 7년차. 품질관리 또는 SCM으로 이동 원함. 이직 전략이 없음.",
     "constraint_hint": "직무 전환 전략 부재"},
    {"domain": "연구개발/R&D", "question_type": "복합질문형", "career_stage": "대학원생", "age": 28,
     "background_hint": "박사 과정 중 산업계 전환 고민. 논문 실적을 어떻게 실무 역량으로 전환할지, 연봉·문화 차이도 궁금.",
     "constraint_hint": "전환 복잡성, 정보 부족"},
    # ── 디자인/UX (4명) ─────────────────────────────
    {"domain": "UX/UI디자인", "question_type": "경험부족형+자료피드백형", "career_stage": "취준생", "age": 24,
     "background_hint": "시각디자인 전공. UX 디자인으로 전환 원함. 포트폴리오를 어떻게 바꿔야 하는지 모름.",
     "constraint_hint": "포트폴리오 전환 전략 없음"},
    {"domain": "UX/UI디자인", "question_type": "가능성판단형", "career_stage": "직장인", "age": 29,
     "background_hint": "그래픽 디자이너 5년차. 테크 기업 Product Designer로 이직 원함. 포트폴리오 경쟁력이 있는지 모름.",
     "constraint_hint": "업종 전환, 포트폴리오 재구성"},
    {"domain": "UX/UI디자인", "question_type": "직무미분화형", "career_stage": "취준생", "age": 23,
     "background_hint": "UX 연구, UI 디자인, 서비스 기획 중 어느 방향으로 가야 할지 모름.",
     "constraint_hint": "직무 미분화"},
    {"domain": "UX/UI디자인", "question_type": "최신전망형", "career_stage": "취준생", "age": 24,
     "background_hint": "AI 디자인 툴이 디자이너를 대체할 것인지 걱정. Figma AI 등장으로 방향을 재고 중.",
     "constraint_hint": "직무 전망 불안"},
    # ── 기타 (4명) ─────────────────────────────────
    {"domain": "공기업/공무원", "question_type": "직무미분화형", "career_stage": "취준생", "age": 25,
     "background_hint": "공기업 vs 대기업 중 어디로 가야 할지 모르겠음. 안정성과 성장성 사이에서 갈등.",
     "constraint_hint": "방향 선택 어려움"},
    {"domain": "유통/무역", "question_type": "커리어로드맵형", "career_stage": "취준생", "age": 23,
     "background_hint": "무역·물류 회사 취직 희망. 어떤 자격증·경험이 필요한지 처음부터 모름.",
     "constraint_hint": "정보 부족"},
    {"domain": "스타트업/창업", "question_type": "직무전환형", "career_stage": "직장인", "age": 31,
     "background_hint": "대기업 5년차. 스타트업 창업 또는 이직 고민. 스타트업 문화와 대기업 차이, 준비 방법이 궁금.",
     "constraint_hint": "리스크 판단, 정보 부족"},
    {"domain": "홍보/PR", "question_type": "가능성판단형+경험부족형", "career_stage": "취준생", "age": 24,
     "background_hint": "언론홍보 지망. 언론학 전공이지만 인턴 경험 없고 포트폴리오 없음.",
     "constraint_hint": "경험 부족, 포트폴리오 없음"},
]

# ─────────────────────────────────────────────────────────────────
# 멘토 시드 (50명)
# ─────────────────────────────────────────────────────────────────

MENTOR_SPECS = [
    {"domain": "금융/투자", "role": "자산운용사 리서치 애널리스트", "exp": 5, "background": "서울대 경제학, 증권사 리서치 인턴 후 자산운용사 RA"},
    {"domain": "금융/투자", "role": "외국계 증권사 IBD 어소시에이트", "exp": 4, "background": "서울대 기계공학(비상경), CFA, IBD 전환 성공"},
    {"domain": "금융/투자", "role": "핀테크 스타트업 CFO", "exp": 8, "background": "연세대 경영, 회계법인 → 대기업 재무 → 스타트업 CFO"},
    {"domain": "금융/투자", "role": "국내 대형 증권사 리테일 PB", "exp": 6, "background": "고려대 경제, 증권사 공채"},
    {"domain": "금융/투자", "role": "사모펀드 심사역", "exp": 7, "background": "포스텍 산업공학, M&A 자문 → PE 심사역"},
    {"domain": "IT개발/백엔드", "role": "네이버 백엔드 시니어 엔지니어", "exp": 7, "background": "카이스트 컴공, 스타트업 → 네이버"},
    {"domain": "IT개발/백엔드", "role": "카카오 백엔드 엔지니어", "exp": 5, "background": "연세대 컴공, 카카오 공채"},
    {"domain": "IT개발/프론트엔드", "role": "토스 프론트엔드 엔지니어", "exp": 4, "background": "한양대 컴공, 부트캠프 수료 후 스타트업 → 토스"},
    {"domain": "IT개발/백엔드", "role": "외국계 테크기업 소프트웨어 엔지니어", "exp": 6, "background": "포스텍 컴공, 삼성 → 구글 코리아"},
    {"domain": "데이터분석/AI", "role": "쿠팡 데이터 사이언티스트", "exp": 5, "background": "서울대 통계학, 스타트업 DA → 쿠팡 DS"},
    {"domain": "데이터분석/AI", "role": "카카오 머신러닝 엔지니어", "exp": 6, "background": "카이스트 AI, 석사 후 카카오"},
    {"domain": "데이터분석/AI", "role": "IT컨설팅 데이터 분석 컨설턴트", "exp": 4, "background": "고려대 통계, 비전공 전환 후 컨설팅 데이터팀"},
    {"domain": "데이터분석/AI", "role": "외국계 테크기업 Product Analyst", "exp": 5, "background": "연세대 경영+통계 복수전공, 스타트업 → 외국계"},
    {"domain": "브랜드마케팅", "role": "나이키코리아 브랜드 마케터", "exp": 6, "background": "이화여대 경영, 대행사 → 나이키"},
    {"domain": "브랜드마케팅", "role": "아모레퍼시픽 브랜드 마케터", "exp": 5, "background": "연세대 생활디자인, 뷰티 브랜드 마케팅"},
    {"domain": "퍼포먼스마케팅", "role": "스타트업 그로스 마케터", "exp": 4, "background": "고려대 경영, 퍼포먼스 마케팅 전문가"},
    {"domain": "MD/유통마케팅", "role": "무신사 MD", "exp": 5, "background": "서울대 의류학, 패션 MD 전문가"},
    {"domain": "브랜드마케팅", "role": "글로벌 FMCG 마케팅 매니저", "exp": 7, "background": "한양대 경영, 외국계 소비재 마케팅"},
    {"domain": "퍼포먼스마케팅", "role": "이커머스 퍼포먼스 마케팅 팀장", "exp": 6, "background": "성균관대 경영, 이커머스 퍼포먼스 전문"},
    {"domain": "MD/유통마케팅", "role": "SSG닷컴 이커머스 MD", "exp": 4, "background": "연세대 경영, 이커머스 MD 커리어"},
    {"domain": "전략기획", "role": "삼성전자 전략기획 파트장", "exp": 9, "background": "서울대 경영, 삼성 공채 → 전략기획"},
    {"domain": "컨설팅", "role": "맥킨지 컨설턴트", "exp": 5, "background": "카이스트 산업공학, MBB 컨설팅"},
    {"domain": "전략기획", "role": "스타트업 COO", "exp": 7, "background": "연세대 경영, 대기업 기획 → 스타트업 COO"},
    {"domain": "컨설팅", "role": "딜로이트 전략 컨설턴트", "exp": 6, "background": "고려대 경영, Big4 컨설팅"},
    {"domain": "전략기획", "role": "LG화학 사업전략팀 과장", "exp": 7, "background": "포스텍 화공, 영업 → 전략기획 이동"},
    {"domain": "HR/인사", "role": "카카오 HRD 매니저", "exp": 6, "background": "이화여대 교육학, HRD 전문가"},
    {"domain": "HR/인사", "role": "삼성물산 채용팀 과장", "exp": 7, "background": "서울대 심리학, 대기업 채용 전문"},
    {"domain": "HR/인사", "role": "스타트업 People Ops 리드", "exp": 5, "background": "연세대 경영, 스타트업 HR 전문"},
    {"domain": "HR/인사", "role": "외국계 기업 HRBP", "exp": 8, "background": "이화여대 경영, 노무·HRBP 전문"},
    {"domain": "HR/인사", "role": "HR테크 스타트업 CPO", "exp": 7, "background": "서울대 산업공학, HRTech + People Analytics"},
    {"domain": "연구개발/R&D", "role": "삼성SDI 배터리 소재 연구원", "exp": 6, "background": "카이스트 화공 석사, 배터리 소재 R&D"},
    {"domain": "생산/품질/제조", "role": "현대자동차 생산관리 차장", "exp": 10, "background": "한양대 기계공학, 생산·SCM 전문"},
    {"domain": "연구개발/R&D", "role": "LG화학 제약연구소 선임연구원", "exp": 7, "background": "연세대 화학 박사, 제약 R&D"},
    {"domain": "생산/품질/제조", "role": "외국계 소비재 SCM 매니저", "exp": 6, "background": "성균관대 산업공학, 품질 → SCM 이동"},
    {"domain": "연구개발/R&D", "role": "반도체 팹 공정 엔지니어", "exp": 5, "background": "POSTECH 전자공학, 반도체 공정 전문"},
    {"domain": "UX/UI디자인", "role": "카카오 Product Designer", "exp": 5, "background": "홍익대 시각디자인, UX 전환 성공"},
    {"domain": "UX/UI디자인", "role": "토스 UX 리서처", "exp": 4, "background": "연세대 심리학, UX 리서치 전문"},
    {"domain": "UX/UI디자인", "role": "외국계 테크기업 Senior UX Designer", "exp": 7, "background": "이화여대 디자인, 그래픽 → Product Design"},
    {"domain": "UX/UI디자인", "role": "스타트업 Head of Design", "exp": 6, "background": "서울대 산업디자인, UX/UI 전반 리드"},
    {"domain": "공기업/공무원", "role": "한국전력 기획팀 대리", "exp": 4, "background": "고려대 행정학, 공기업 공채 성공"},
    {"domain": "유통/무역", "role": "대한무역투자진흥공사 무역전문가", "exp": 8, "background": "한국외대 무역학, 무역·물류 전문"},
    {"domain": "스타트업/창업", "role": "시리즈B 스타트업 CMO", "exp": 7, "background": "연세대 경영, 대기업 → 스타트업 창업"},
    {"domain": "홍보/PR", "role": "대기업 홍보팀 차장", "exp": 9, "background": "서울대 언론정보학, 언론사 → 홍보"},
    {"domain": "금융/투자", "role": "벤처캐피탈 심사역", "exp": 5, "background": "카이스트 경영, 스타트업 → VC"},
    {"domain": "데이터분석/AI", "role": "네이버 AI 리서처", "exp": 6, "background": "서울대 AI 박사, LLM 연구"},
    {"domain": "IT개발/백엔드", "role": "쿠팡 테크 리드", "exp": 8, "background": "포스텍 컴공, 대규모 분산 시스템"},
    {"domain": "브랜드마케팅", "role": "스타트업 마케팅 디렉터", "exp": 8, "background": "이화여대 경영, 스타트업 브랜딩·바이럴"},
    {"domain": "전략기획", "role": "글로벌 테크기업 BizDev 매니저", "exp": 6, "background": "연세대 경영, 전략 → 글로벌 사업개발"},
    {"domain": "컨설팅", "role": "부티크 컨설팅 파트너", "exp": 12, "background": "서울대 경영, MBB → 부티크 창업"},
    {"domain": "HR/인사", "role": "대기업 노무사 겸 HR팀장", "exp": 10, "background": "고려대 법학, 공인노무사 취득 후 대기업 HR"},
]


# ─────────────────────────────────────────────────────────────────
# LLM 생성 함수
# ─────────────────────────────────────────────────────────────────

MENTEE_SYSTEM = """너는 한국 취업·진로 멘토링 플랫폼의 가상 멘티 페르소나를 생성하는 AI다.
반드시 아래 JSON 스키마를 준수하라:
{
  "mentee_info": {"name": "string(한국 실명)", "gender": "남|여", "age": int},
  "background": {"school": "string(실명)", "major": "string", "grade": "string|null", "status": "대학생|취준생|대학원생|직장인"},
  "career_profile": {"current_status": "string(현재 상태 1줄)", "years_experience": int, "current_role": "string(현 직함 또는 없으면 빈 문자열)"},
  "interest_profile": {"interest_domain": ["string"], "desired_roles": ["string(목표 직무 1~3개)"]},
  "transition_profile": {
    "source_role": "string(현재/이전 직무, 없으면 빈 문자열)",
    "target_role": "string(목표 직무)",
    "transferable_skills": ["string"],
    "target_domain_candidates": ["string"],
    "bridge_hypothesis": "string(전환 가설 1문장, 학생이면 빈 문자열)"
  },
  "matching_profile": {
    "matching_summary_text": "string(검색·매칭용 요약, 3~5문장, 배경·고민·목표·병목 포함)",
    "be_go": "string(이 멘티가 어떤 멘토에게 맞는지 1~2문장)"
  }
}
JSON 외 다른 텍스트 출력 금지."""

MENTOR_SYSTEM = """너는 한국 취업·진로 멘토링 플랫폼의 가상 멘토 페르소나를 생성하는 AI다.
반드시 아래 JSON 스키마를 준수하라:
{
  "mentor_info": {"name": "string(한국 실명)", "gender": "남|여", "age": int},
  "background": {"school": "string(실명)", "major": "string", "grade": null, "status": "직장인"},
  "current_role": "string(현재 직함, 회사명 포함)",
  "years_of_experience": int,
  "matching_summary_text": "string(검색·매칭용 자기소개, 4~6문장, 경력·전문성·멘토링 강점 포함)",
  "be_go": "string(이 멘토가 어떤 멘티에게 적합한지 1~2문장)"
}
JSON 외 다른 텍스트 출력 금지."""


def call_llm(system: str, user: str, model: str = "gpt-4o") -> dict:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": user}],
        temperature=0.85,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def generate_mentee(spec: dict) -> dict:
    user_msg = (
        f"도메인: {spec['domain']}\n"
        f"질문유형: {spec['question_type']}\n"
        f"커리어단계: {spec['career_stage']} (나이 {spec['age']}세)\n"
        f"배경힌트: {spec['background_hint']}\n"
        f"제약힌트: {spec['constraint_hint']}\n"
        "위 정보를 반영한 멘티 페르소나를 JSON으로 출력하라."
    )
    data = call_llm(MENTEE_SYSTEM, user_msg)
    ts = now_str()
    return {
        "mentee_id":            new_id("mt_"),
        "mentee_info":          data.get("mentee_info", {}),
        "background":           data.get("background", {}),
        "career_profile":       data.get("career_profile", {}),
        "interest_profile":     data.get("interest_profile", {}),
        "transition_profile":   data.get("transition_profile", {}),
        "persistent_bottlenecks": [],
        "matching_profile":     data.get("matching_profile", {}),
        "session_stats":        {"total_sessions": 0, "last_session_at": None},
        "active":               True,
        "created_at":           ts,
        "updated_at":           ts,
    }


def generate_mentor(spec: dict) -> dict:
    user_msg = (
        f"도메인: {spec['domain']}\n"
        f"현재직무: {spec['role']} (경력 {spec['exp']}년)\n"
        f"배경: {spec['background']}\n"
        "위 정보를 반영한 멘토 페르소나를 JSON으로 출력하라."
    )
    data = call_llm(MENTOR_SYSTEM, user_msg)
    return {
        "mentor_id":             new_id("mr_"),
        "mentor_info":           data.get("mentor_info", {}),
        "background":            data.get("background", {}),
        "current_role":          data.get("current_role", spec["role"]),
        "years_of_experience":   data.get("years_of_experience", spec["exp"]),
        "total_questions":       0,
        "matching_summary_text": data.get("matching_summary_text", ""),
        "be_go":                 data.get("be_go", ""),
        "active":                True,
        "created_at":            now_str(),
    }


# ─────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────

def run(only: str | None, replace: bool) -> None:
    if only in (None, "mentee"):
        print(f"=== 멘티 {len(MENTEE_SPECS)}명 생성 중 ===")
        mt_path = DB_DIR / "mentees.json"
        db = json.loads(mt_path.read_text(encoding="utf-8"))
        if replace:
            db["records"] = []
        records = []
        for i, spec in enumerate(MENTEE_SPECS, 1):
            try:
                rec = generate_mentee(spec)
                records.append(rec)
                print(f"  [{i:02d}/{len(MENTEE_SPECS)}] {rec['mentee_info'].get('name','?')}"
                      f" — {spec['domain']} / {spec['question_type']}")
            except Exception as e:
                print(f"  [{i:02d}] ERROR: {e}")
        db["records"].extend(records)
        mt_path.write_text(json.dumps(db, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  → mentees.json 저장 완료 ({len(records)}명)")

    if only in (None, "mentor"):
        print(f"\n=== 멘토 {len(MENTOR_SPECS)}명 생성 중 ===")
        mr_path = DB_DIR / "mentors.json"
        # mentors.json 손상 시 빈 구조로 초기화
        try:
            db = json.loads(mr_path.read_text(encoding="utf-8"))
        except Exception:
            db = {"schema_name": "mentors", "description": "멘토 기본 정보 DB", "records": []}
        if replace:
            db["records"] = []
        records = []
        for i, spec in enumerate(MENTOR_SPECS, 1):
            try:
                rec = generate_mentor(spec)
                records.append(rec)
                print(f"  [{i:02d}/{len(MENTOR_SPECS)}] {rec['mentor_info'].get('name','?')}"
                      f" — {rec['current_role']}")
            except Exception as e:
                print(f"  [{i:02d}] ERROR: {e}")
        db["records"].extend(records)
        mr_path.write_text(json.dumps(db, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  → mentors.json 저장 완료 ({len(records)}명)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["mentee", "mentor"], default=None)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    run(args.only, args.replace)
