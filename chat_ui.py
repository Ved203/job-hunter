import os
import streamlit as st
import sys
from pathlib import Path

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

from job_search import run_job_search
from query_agent import (
    SCHEMA,
    intent_chain,
    greeting_chain,
    sql_chain,
    summary_chain,
    extract_search_parameters,
    fetch_jobs_for_search,
    format_jobs_for_context,
    run_collected_search,
    get_total_job_count,
    is_safe_query,
    clean_sql,
    run_query,
    MAX_ROWS_FOR_LLM,
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(page_title="Smart Job Hunter", page_icon="🧭", layout="wide")

# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "collecting_search" not in st.session_state:
    st.session_state.collecting_search = False
if "awaiting_date_range" not in st.session_state:
    st.session_state.awaiting_date_range = False
if "search_state" not in st.session_state:
    st.session_state.search_state = {"role_or_skills": None, "experience": None, "location": None}
if "last_jobs" not in st.session_state:
    st.session_state.last_jobs = []
if "last_excel_path" not in st.session_state:
    st.session_state.last_excel_path = None
if "current_batch_id" not in st.session_state:
    st.session_state.current_batch_id = None


def reset_search_state():
    st.session_state.collecting_search = False
    st.session_state.awaiting_date_range = False
    st.session_state.search_state = {"role_or_skills": None, "experience": None, "location": None}


def get_conversation_history(limit=10):
    messages = st.session_state.messages[-limit:]
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)


def next_missing_field():
    state = st.session_state.search_state
    if not state["role_or_skills"]:
        return "role_or_skills", "What role or skills are you looking for?"
    if not state["experience"]:
        return "experience", "What's your experience level? (e.g. fresher, 1 year, 2-3 years)"
    if not state["location"]:
        return "location", "Which location? (e.g. Pune, Mumbai, Remote)"
    return None, None


def format_search_results(jobs):
    if not jobs:
        return "No matching jobs were found."

    lines = []
    for index, job in enumerate(jobs, start=1):
        title = job.get("title", "Unknown title")
        company = job.get("company_name", "Unknown company")
        location = job.get("location", "Unknown location")
        seniority = job.get("seniority_level", "Not specified")
        employment_type = job.get("employment_type", "Not specified")
        link = job.get("url")
        actual_experience = job.get("actual_experience")
        mismatch = job.get("experience_mismatch")
        applicants = job.get("applicants_text")
        salary = job.get("salary_text")
        ghost_risk = job.get("ghost_job_risk")

        exp_line = f"🕒 **LinkedIn tag:** {seniority}"
        if actual_experience:
            exp_line += f"  |  **Description says:** {actual_experience}"
            if mismatch:
                exp_line += "  ⚠️ **Mismatch detected**"

        extra_bits = []
        if applicants:
            extra_bits.append(f"👥 {applicants}")
        if salary:
            extra_bits.append(f"💰 {salary}")
        if ghost_risk and ghost_risk.lower() not in ["unknown", "low"]:
            extra_bits.append(f"⚠️ Ghost job risk: {ghost_risk}")
        extra_line = "  |  ".join(extra_bits)

        card = (
            f"**{index}. {title}**\n"
            f"🏢 **Company:** {company}\n"
            f"📍 **Location:** {location}\n"
            f"{exp_line}\n"
            f"💼 **Type:** {employment_type}"
        )
        if extra_line:
            card += f"\n{extra_line}"
        card += f"\n🔗 **Job Link:** [Open Job]({link})"

        lines.append(card)

    return "\n\n---\n\n".join(lines)


def handle_question(user_message):
    if get_total_job_count() == 0:
        return "You haven't searched for any jobs yet! Try saying \"find me nursing jobs in Mumbai\" to get started."

    history = get_conversation_history()
    current_jobs = format_jobs_for_context(st.session_state.last_jobs)
    current_batch_id = st.session_state.current_batch_id or "none"

    sql = clean_sql(sql_chain.invoke({
        "schema": SCHEMA,
        "history": history,
        "current_jobs": current_jobs,
        "current_batch_id": current_batch_id,
        "question": user_message
    }))

    print("\n[Generated SQL]")
    print(sql)

    if not is_safe_query(sql):
        return "Sorry, I couldn't safely process that request."

    columns, rows = run_query(sql)

    if not rows:
        return "No matching jobs found in your latest search.\n\nTry asking about \"all my searches\", or search for something new."

    jobs = [dict(zip(columns, row)) for row in rows]
    st.session_state.last_jobs = jobs

    total_count = len(jobs)
    note = ""
    if total_count > 10:
        note = f"\n\n*There are {total_count} total matches. Showing the top 10.*"
        jobs_to_show = jobs[:10]
    else:
        jobs_to_show = jobs

    return format_search_results(jobs_to_show) + note


def handle_message(user_message):
    user_message = user_message.strip()
    if not user_message:
        return "Please enter a message."

    history = get_conversation_history()

    if st.session_state.collecting_search:
        if user_message.lower() in ["cancel", "stop", "nevermind"]:
            reset_search_state()
            return "❌ Search cancelled."

        extracted = extract_search_parameters(
            message=user_message, history=history, current_state=st.session_state.search_state
        )
        state = st.session_state.search_state
        for field in ["role_or_skills", "experience", "location"]:
            value = extracted.get(field)
            if value:
                state[field] = value

        field, question = next_missing_field()
        if field:
            return question

        st.session_state.collecting_search = False
        st.session_state.awaiting_date_range = True
        return "One last thing — how far back should I search? Pick an option below 👇"

    intent = intent_chain.invoke({"history": history, "message": user_message}).strip().upper()
    print(f"\n[Detected intent]: {intent}\n")

    if intent == "GREETING":
        return greeting_chain.invoke({"history": history, "message": user_message})

    if intent == "SEARCH":
        reset_search_state()
        extracted = extract_search_parameters(
            message=user_message, history=history, current_state=st.session_state.search_state
        )
        state = st.session_state.search_state
        for field in ["role_or_skills", "experience", "location"]:
            value = extracted.get(field)
            if value:
                state[field] = value

        field, question = next_missing_field()
        if field:
            st.session_state.collecting_search = True
            return "🔎 Let's find the right jobs.\n\n" + question

        st.session_state.awaiting_date_range = True
        return "One last thing — how far back should I search? Pick an option below 👇"

    if intent == "QUESTION":
        return handle_question(user_message)

    if intent == "APPLY":
        return "The apply feature is being rebuilt right now — for the moment, please use the job links directly to apply."

    return (
        "I'm not sure how to help with that.\n\n"
        "Try:\n"
        "- Find nursing jobs in Mumbai\n"
        "- Show the latest jobs\n"
        "- Show remote ones\n"
        "- Which companies are hiring?"
    )


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 38px;
        font-weight: 800;
        margin-bottom: 2px;
        background: linear-gradient(90deg, #4F8BF9, #A855F7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {
        color: #999;
        font-size: 16px;
        margin-bottom: 22px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("🧭 Smart Job Hunter")
    st.caption("Search any role, any industry, any location.")
    st.markdown("---")

    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.messages = []
        reset_search_state()
        st.session_state.last_jobs = []
        st.session_state.last_excel_path = None
        st.session_state.current_batch_id = None
        st.rerun()

    st.markdown("---")
    st.subheader("How it works")
    st.markdown(
        """
        **1. 🔎 Search** — tell me a role, experience level, and location.

        **2. 📊 Ask** — filter, compare, or explore what's saved.

        **3. 📁 Export** — download every search as its own Excel file.
        """
    )

    st.markdown("---")
    if st.session_state.last_jobs:
        st.metric("Jobs in current view", len(st.session_state.last_jobs))

    # ---------------- DOWNLOAD BUTTON (moved here, in sidebar) ----------------
    st.markdown("---")
    if st.session_state.last_excel_path and os.path.exists(st.session_state.last_excel_path):
        with open(st.session_state.last_excel_path, "rb") as f:
            st.download_button(
                label="📥 Download latest results",
                data=f,
                file_name=os.path.basename(st.session_state.last_excel_path),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        st.caption(os.path.basename(st.session_state.last_excel_path))
    else:
        st.caption("📁 No search results yet — the Excel file will appear here after your first search.")

    st.markdown("---")
    st.caption("Works for any role — tech, healthcare, marketing, finance, and more.")

# ============================================================
# HEADER
# ============================================================

st.markdown('<div class="main-title">🧭 Smart Job Hunter</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Search, filter, and track job openings across any field — powered by an AI agent.</div>',
    unsafe_allow_html=True
)

# ============================================================
# WELCOME SCREEN
# ============================================================

if not st.session_state.messages:
    st.info("👋 Tell me what job you're looking for, or ask about jobs already saved.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("🏥 **Find nursing jobs in Mumbai**")
    with col2:
        st.write("📈 **Marketing roles in Pune**")
    with col3:
        st.write("💻 **Software engineer jobs, remote**")

# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ============================================================
# DATE RANGE BUTTONS (shown only when needed)
# ============================================================

if st.session_state.awaiting_date_range:
    st.write("**Choose a date range:**")
    col1, col2, col3 = st.columns(3)

    date_choice = None
    with col1:
        if st.button("🕐 Last 24 hours", use_container_width=True):
            date_choice = "past24Hours"
    with col2:
        if st.button("📅 Last week", use_container_width=True):
            date_choice = "pastWeek"
    with col3:
        if st.button("🗓️ Last month", use_container_width=True):
            date_choice = "pastMonth"

    if date_choice:
        label_map = {"past24Hours": "Last 24 hours", "pastWeek": "Last week", "pastMonth": "Last month"}
        st.session_state.messages.append({"role": "user", "content": f"(selected: {label_map[date_choice]})"})

        with st.spinner("Searching and verifying job details... this may take a minute."):
            inserted, skipped, jobs, excel_path, batch_id = run_collected_search(
                st.session_state.search_state,
                run_job_search,
                date_posted=date_choice
            )

        reset_search_state()
        st.session_state.last_jobs = jobs
        st.session_state.last_excel_path = excel_path
        st.session_state.current_batch_id = batch_id

        response = f"✅ Search complete! Added **{inserted}** new job(s) (skipped {skipped} duplicates). *(Excel file ready in the sidebar →)*"
        if jobs:
            response += "\n\n### Matching jobs\n\n" + format_search_results(jobs)
        else:
            response += "\n\nNo matching jobs found yet — try broadening your search."

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

# ============================================================
# CHAT INPUT
# ============================================================

user_input = st.chat_input(
    "Ask about jobs...",
    disabled=st.session_state.awaiting_date_range
)

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    try:
        response = handle_message(user_input)
    except Exception as e:
        response = f"❌ Error: {e}"
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()