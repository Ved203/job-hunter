import os
import sqlite3
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from apify_client import ApifyClient

load_dotenv()


def run_job_search(keywords=None, locations=None, date_posted="past24Hours"):
    """Runs a fresh Apify search, saves to SQLite, and exports a timestamped Excel file.
    Returns (inserted, skipped, excel_path, newly_added_ids, batch_id)."""
    API_TOKEN = os.getenv("APIFY_API_TOKEN")
    client = ApifyClient(API_TOKEN)

    batch_id = datetime.now().strftime("%Y%m%d%H%M%S")  # unique ID for this search run

    run_input = {
        "keywords": keywords or [
            "Generative AI Engineer", "AI Engineer", "Junior AI Engineer",
            "Machine Learning Engineer", "GenAI Developer", "AI/ML Intern"
        ],
        "locations": locations or [
            "Pune, India", "Pimpri-Chinchwad, India", "Mumbai, India", "Navi Mumbai, India"
        ],
        "datePosted": date_posted,
        "experienceLevel": ["Entry level"],
        "maxJobsPerSearch": 100,
        "sortBy": "date"
    }

    conn = sqlite3.connect("jobs.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS jobs (
        id TEXT PRIMARY KEY,
        title TEXT,
        company_name TEXT,
        location TEXT,
        seniority_level TEXT,
        employment_type TEXT,
        posted_text TEXT,
        description_text TEXT,
        url TEXT UNIQUE,
        applicants_text TEXT,
        salary_text TEXT,
        ghost_job_risk TEXT,
        company_website TEXT,
        actual_experience TEXT,
        experience_mismatch INTEGER DEFAULT 0,
        search_batch_id TEXT
    )
    """)
    conn.commit()

    print("Searching for jobs... this may take a minute.")
    run = client.actor("kamerozkan/linkedin-jobs-scraper").call(run_input=run_input)
    dataset_items = client.dataset(run.default_dataset_id).list_items().items

    inserted, skipped = 0, 0
    newly_added_ids = []
    all_matched_ids = []  # every job matched THIS search, whether new or already existed

    for job in dataset_items:
        job_id = job.get("id")
        all_matched_ids.append(job_id)
        try:
            cursor.execute("""
                INSERT INTO jobs (
                    id, title, company_name, location, seniority_level, employment_type,
                    posted_text, description_text, url, applicants_text, salary_text,
                    ghost_job_risk, company_website, search_batch_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job_id,
                job.get("title"),
                job.get("companyName"),
                job.get("location"),
                job.get("seniorityLevel"),
                job.get("employmentType"),
                job.get("postedText"),
                job.get("descriptionText", ""),
                job.get("url"),
                job.get("applicantsText"),
                job.get("salaryText"),
                job.get("ghostJobRisk"),
                job.get("companyWebsite"),
                batch_id
            ))
            inserted += 1
            newly_added_ids.append(job_id)
        except sqlite3.IntegrityError:
            skipped += 1
            # Job already existed from a PREVIOUS search — re-tag it with the current batch
            # so it shows up in "this search's" results too, since it matched again
            cursor.execute("UPDATE jobs SET search_batch_id = ? WHERE id = ?", (batch_id, job_id))

    conn.commit()

    os.makedirs("exports", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    excel_path = f"exports/jobs_export_{timestamp}.xlsx"

    if all_matched_ids:
        placeholders = ",".join("?" * len(all_matched_ids))
        df = pd.read_sql_query(
            f"SELECT * FROM jobs WHERE id IN ({placeholders}) ORDER BY posted_text DESC",
            conn, params=all_matched_ids
        )
    else:
        df = pd.read_sql_query("SELECT * FROM jobs WHERE 1=0", conn)

    df.to_excel(excel_path, index=False)
    print(f"Excel file created: {excel_path} ({len(df)} jobs from this search)")

    conn.close()
    return inserted, skipped, excel_path, all_matched_ids, batch_id


if __name__ == "__main__":
    inserted, skipped, excel_path, _, batch_id = run_job_search()
    print(f"Done! Inserted {inserted} new jobs, skipped {skipped} duplicates. Batch: {batch_id}. Excel: {excel_path}")