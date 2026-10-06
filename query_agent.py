import os
import sqlite3
import json

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


# ============================================================
# LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)


# ============================================================
# DATABASE SCHEMA  (must match job_search.py's CREATE TABLE)
# ============================================================

SCHEMA = """
Table: jobs

Columns:
- id (TEXT)
- title (TEXT)
- company_name (TEXT)
- location (TEXT)
- seniority_level (TEXT)
- employment_type (TEXT)
- posted_text (TEXT)
- description_text (TEXT)
- url (TEXT)
- applicants_text (TEXT)
- salary_text (TEXT)
- ghost_job_risk (TEXT)
- company_website (TEXT)
- actual_experience (TEXT)
- experience_mismatch (INTEGER)
- search_batch_id (TEXT)
"""


MIN_GOOD_MATCHES = 3
MAX_ROWS_FOR_LLM = 30

DB_PATH = "jobs.db"


# ============================================================
# INTENT CLASSIFIER
# ============================================================

intent_prompt = ChatPromptTemplate.from_template(
    """
You are the router for a job-search assistant.

Classify the CURRENT user message into exactly one category.

GREETING:
- hello, hi, thanks, casual conversation, small talk

SEARCH:
- user wants to perform a NEW job search with criteria not yet searched
- user asks to find, search, look for a NEW role/location not already discussed

QUESTION:
- user is asking about jobs already stored in the database
- user asks to show, filter, count, compare, sort, or explain existing jobs
- user refers to previous results such as "these jobs", "those jobs",
  "the first one", "remote ones", "companies mentioned earlier"
- "which companies are hiring" = QUESTION (asking about existing data)
- "show me the jobs" = QUESTION
- "give me links" = QUESTION

APPLY:
- user wants to apply to a specific job
- user says apply, open application, start application, apply to first/second job, etc.

EXAMPLES:
"I want to search for jobs" -> SEARCH
"find me AI jobs in Mumbai" -> SEARCH
"which companies are hiring?" -> QUESTION
"show me remote jobs" -> QUESTION
"give me top 5 jobs" -> QUESTION
"give me the link of 20 jobs" -> QUESTION

IMPORTANT:
Use conversation history to understand the current message.

Previous conversation:
{history}

Current user message:
{message}

Return ONLY one word:
GREETING
SEARCH
QUESTION
APPLY
"""
)

intent_chain = intent_prompt | llm | StrOutputParser()


# ============================================================
# GREETING
# ============================================================

greeting_prompt = ChatPromptTemplate.from_template(
    """
You are a friendly AI job-search assistant.

Respond naturally and briefly.

You can help the user:
- find new jobs
- answer questions about saved jobs
- filter or analyze jobs
- start an application

Previous conversation:
{history}

Current message:
{message}
"""
)

greeting_chain = greeting_prompt | llm | StrOutputParser()


# ============================================================
# SEARCH PARAMETER EXTRACTION
# ============================================================

search_params_prompt = ChatPromptTemplate.from_template(
    """
Extract structured job-search information from the current conversation.

Previous conversation:
{history}

Current search state:
{current_state}

Current user message:
{message}

Return ONLY valid JSON in this exact format:

{{
    "role_or_skills": null,
    "experience": null,
    "location": null,
    "keywords": [],
    "locations": []
}}

Rules:
1. Extract information explicitly provided by the user.
2. Use previous conversation when the user is answering a question from the assistant.
3. Do not invent missing information.
4. If role/skill is known, put it in role_or_skills.
5. If experience is known, put it in experience.
6. If location is known, put it in location.
7. keywords should contain useful search keywords.
8. locations should contain useful location values.
9. If a field is already present in current_state and the user is not changing it, preserve it.
10. If the user provides more than one piece of information in one sentence, extract all of them.
11. Return raw JSON only.
"""
)

search_params_chain = search_params_prompt | llm | StrOutputParser()


# ============================================================
# SQL GENERATION
# ============================================================
sql_prompt = ChatPromptTemplate.from_template(
    """
You are a SQLite SQL generator for a job-search chatbot.

Database schema:
{schema}

Previous conversation:
{history}

Currently remembered/displayed jobs:
{current_jobs}

Current active search batch ID: {current_batch_id}

Current user request:
{question}

Convert the user's current request into ONE SQLite SELECT query.

Rules:
1. Use SELECT *.
2. Only SELECT statements are allowed.
3. Never use INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, PRAGMA, ATTACH, or other write operations.
4. Use LIKE for text matching — extract only the actual keyword, ignore filler words like "database", "details", "give me", "role".
5. For text values use: LIKE '%value%'
6. LIMIT 5 by default.
7. If the user asks for all/everything, do not add LIMIT 5.
8. If the user asks for a specific number, use that number.
9. Order by posted_text DESC by default.
10. Use conversation history to resolve references such as "these jobs", "those jobs", "first one", "second one", "remote ones", "companies mentioned earlier".
11. If currently displayed jobs identify a specific record, use its id when appropriate.
12. UNLESS the user explicitly asks about "all my searches", "everything I've ever searched", or "my full history", ALWAYS add: WHERE search_batch_id = '{current_batch_id}' (combined with AND if other conditions exist).
13. Output ONLY raw SQL.

SQL:
"""
)

sql_chain = sql_prompt | llm | StrOutputParser()


# ============================================================
# APPLY TARGET
# ============================================================

apply_target_prompt = ChatPromptTemplate.from_template(
    """
You are selecting job records for an application request.

Database schema:
{schema}

Previous conversation:
{history}

Currently remembered/displayed jobs:
{current_jobs}

Current user request:
{message}

Generate ONE SQLite SELECT * query that finds the job(s) the user wants to apply to.

Rules:
1. SELECT * only.
2. Use LIKE '%value%' for text matching.
3. Only SELECT statements.
4. Never modify the database.
5. Understand references such as: first job, second job, that AI Engineer job, the job at ABC, the one we discussed earlier.
6. Use remembered/displayed jobs when possible.
7. Output ONLY raw SQL.
"""
)

apply_target_chain = apply_target_prompt | llm | StrOutputParser()


# ============================================================
# EXPERIENCE VERIFICATION (checks description vs LinkedIn tag)
# ============================================================

experience_check_prompt = ChatPromptTemplate.from_template(
    """
Read this job description and determine the ACTUAL experience requirement stated in the text.

Job description:
{description}

LinkedIn's own seniority tag for this job: {linkedin_seniority}

Return ONLY valid JSON in this exact format:
{{
    "actual_experience": "0-1 years",
    "mismatch": true
}}

Rules:
1. actual_experience should be a short phrase like "0-1 years", "2-5 years", "Fresher", "Not stated" (if the description genuinely doesn't mention it).
2. mismatch is true if LinkedIn's tag clearly contradicts what the description says (e.g. LinkedIn says "Entry level" but description says "5+ years required").
3. mismatch is false if they roughly agree, or if the description doesn't state anything specific enough to compare.
4. Return raw JSON only, no explanation.
"""
)

experience_check_chain = experience_check_prompt | llm | StrOutputParser()


# ============================================================
# SUMMARY
# ============================================================

summary_prompt = ChatPromptTemplate.from_template(
    """
The user asked:
{question}

Here are the matching database results:
{results}

Give a short, natural answer.
For each job, include: Title, Company, Location.
Put each job on its own line.
Do not claim that these are all jobs unless the query actually returned all matching jobs.
Do not invent information.
"""
)

summary_chain = summary_prompt | llm | StrOutputParser()


# ============================================================
# SQL SAFETY
# ============================================================

def is_safe_query(sql):
    sql_upper = sql.upper().strip()
    forbidden = [
        "DROP", "DELETE", "UPDATE", "INSERT", "ALTER",
        "TRUNCATE", "PRAGMA", "ATTACH", "DETACH", "REPLACE"
    ]
    return sql_upper.startswith("SELECT") and not any(word in sql_upper for word in forbidden)


def clean_sql(raw_sql):
    return raw_sql.replace("```sql", "").replace("```", "").strip()


def clean_json(raw_text):
    return raw_text.replace("```json", "").replace("```", "").strip()

def run_query(sql):
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        columns = [d[0] for d in cursor.description]
        rows = cursor.fetchall()
        return columns, rows
    finally:
        conn.close()


def get_total_job_count():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM jobs")
    count = cursor.fetchone()[0]
    conn.close()
    return count

def rows_to_dicts(columns, rows):
    return [dict(zip(columns, row)) for row in rows]


def format_jobs_for_context(jobs):
    if not jobs:
        return "No currently remembered jobs."

    lines = []
    for index, job in enumerate(jobs[:20], start=1):
        lines.append(f"""JOB {index}
id: {job.get("id")}
title: {job.get("title")}
company: {job.get("company_name")}
location: {job.get("location")}
posted_date: {job.get("posted_text")}
seniority: {job.get("seniority_level")}
employment_type: {job.get("employment_type")}
link: {job.get("url")}""".strip())

    return "\n\n".join(lines)


def extract_search_parameters(message, history="", current_state=None):
    if current_state is None:
        current_state = {"role_or_skills": None, "experience": None, "location": None}

    try:
        raw = search_params_chain.invoke({
            "history": history,
            "current_state": json.dumps(current_state, ensure_ascii=False),
            "message": message
        })
        raw = clean_json(raw)
        result = json.loads(raw)

        if not isinstance(result, dict):
            return {"role_or_skills": None, "experience": None, "location": None, "keywords": [], "locations": []}

        return result

    except Exception as e:
        print(f"[Search parameter extraction error]: {e}")
        return {"role_or_skills": None, "experience": None, "location": None, "keywords": [], "locations": []}


def verify_experience_for_new_jobs(newly_added_ids):
    """Checks each newly added job's description against its LinkedIn seniority tag, updates the DB."""
    if not newly_added_ids:
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    placeholders = ",".join("?" * len(newly_added_ids))
    cursor.execute(
        f"SELECT id, description_text, seniority_level FROM jobs WHERE id IN ({placeholders})",
        newly_added_ids
    )
    rows = cursor.fetchall()

    for job_id, description, seniority in rows:
        if not description:
            continue

        try:
            raw = experience_check_chain.invoke({
                "description": description[:2000],
                "linkedin_seniority": seniority or "Not specified"
            })
            raw = clean_json(raw)
            result = json.loads(raw)

            actual_experience = result.get("actual_experience", "Not stated")
            mismatch = 1 if result.get("mismatch") else 0

            cursor.execute(
                "UPDATE jobs SET actual_experience = ?, experience_mismatch = ? WHERE id = ?",
                (actual_experience, mismatch, job_id)
            )
        except Exception as e:
            print(f"[Experience check error for job {job_id}]: {e}")
            continue

    conn.commit()
    conn.close()


def fetch_jobs_for_search(keywords=None, locations=None, limit=20):
    keywords = keywords or []
    locations = locations or []

    conditions = []
    params = []

    if keywords:
        keyword_conditions = []
        for keyword in keywords:
            keyword_conditions.append("(LOWER(title) LIKE LOWER(?) OR LOWER(company_name) LIKE LOWER(?))")
            value = f"%{keyword}%"
            params.extend([value, value])
        conditions.append("(" + " OR ".join(keyword_conditions) + ")")

    if locations:
        location_conditions = []
        for location in locations:
            location_conditions.append("LOWER(location) LIKE LOWER(?)")
            params.append(f"%{location}%")
        conditions.append("(" + " OR ".join(location_conditions) + ")")
    sql = ("SELECT id, title, company_name, location, posted_text, seniority_level, "
       "employment_type, url, actual_experience, experience_mismatch, "
       "applicants_text, salary_text, ghost_job_risk, company_website FROM jobs")
    
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)

    sql += " ORDER BY posted_text DESC LIMIT ?"
    params.append(limit)

    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute(sql, params)
        columns = [d[0] for d in cursor.description]
        rows = cursor.fetchall()
        return rows_to_dicts(columns, rows)
    finally:
        conn.close()


def run_collected_search(search_state, run_job_search_fn, date_posted="past24Hours"):
    role = search_state.get("role_or_skills")
    location = search_state.get("location")

    keywords = [role] if role else []
    locations = [loc.strip() for loc in location.split(",")] if location else []

    inserted, skipped, excel_path, matched_ids, batch_id = run_job_search_fn(
        keywords=keywords, locations=locations, date_posted=date_posted
    )

    verify_experience_for_new_jobs(matched_ids)

    jobs = fetch_jobs_for_search(keywords=keywords, locations=locations, limit=10)

    return inserted, skipped, jobs, excel_path, batch_id
