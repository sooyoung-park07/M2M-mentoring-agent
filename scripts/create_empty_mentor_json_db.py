import json
from pathlib import Path


# ============================================================
# 0. 경로 설정
# ============================================================

# 현재 파일 위치:
# 프로젝트2팀/data_db/scripts/create_empty_mentor_json_db.py
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DB_DIR = SCRIPT_DIR.parent

# JSON DB 저장 위치:
# 프로젝트2팀/data_db/json_db/
JSON_DB_DIR = DATA_DB_DIR / "json_db"
JSON_DB_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. 빈 JSON DB 스키마 정의
# ============================================================

mentors_db = {
    "schema_name": "mentors",
    "description": "멘토 기본 정보 DB",
    "fields": {
        "mentor_id": "string",
        "mentor_info": {
            "name": "string",
            "gender": "string",
            "age": "int"
        },
        "background": {
            "school": "string",
            "major": "string",
            "grade": "string or null",
            "status": "string"
        },
        "current_role": "string",
        "years_of_experience": "int",
        "total_questions": "int",
        "matching_summary_text": "string",
        "be_go": "string",
        "active": "bool"
    },
    "records": []
}


mentor_experiences_db = {
    "schema_name": "mentor_experiences",
    "description": "멘토 경험 정보 DB",
    "fields": {
        "experience_id": "string",
        "mentor_id": "string",
        "experience_type": "project | internship | work | club | course | certification | award | education | etc",
        "title": "string",
        "description": "string",
        "organization": "string",
        "start_date": "YYYY-MM",
        "end_date": "YYYY-MM or present",
        "role": "string",
        "key_skills": "string",
        "tools": "string"
    },
    "records": []
}


mentor_answers_db = {
    "schema_name": "mentor_answers",
    "description": "멘토 답변 정보 DB",
    "fields": {
        "answer_id": "string",
        "mentor_id": "string",
        "question_content": "string",
        "answer_content": "string",
        "answer_summarize": "string"
    },
    "records": []
}


# ============================================================
# 2. 저장 함수
# ============================================================

def save_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ============================================================
# 3. 메인 실행
# ============================================================

def main():
    save_json(JSON_DB_DIR / "mentors.json", mentors_db)
    save_json(JSON_DB_DIR / "mentor_experiences.json", mentor_experiences_db)
    save_json(JSON_DB_DIR / "mentor_answers.json", mentor_answers_db)

    print("빈 JSON 멘토 DB 생성 완료")
    print(f"저장 위치: {JSON_DB_DIR}")
    print("- mentors.json")
    print("- mentor_experiences.json")
    print("- mentor_answers.json")


if __name__ == "__main__":
    main()