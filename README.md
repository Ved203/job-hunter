# 🤖 GenAI Job Hunter

An AI-powered job search application that helps users find and explore relevant job opportunities using automated job data collection, natural language queries, and AI-powered search.

The project combines job data collection, data processing, database storage, and Generative AI to create a more convenient way to search for job opportunities.

---

## 📌 Project Overview

Searching for jobs manually across different platforms can be time-consuming.

A job seeker may need to:

- Search for jobs on different platforms
- Apply multiple filters
- Check job descriptions
- Identify required skills
- Compare different opportunities
- Repeat the same process every day

The goal of **GenAI Job Hunter** is to simplify this process by bringing job data and AI-powered search into a single application.

Users can interact with the system using natural-language queries instead of manually applying multiple filters.

For example:

> Find Python developer jobs in Pune

or

> Find GenAI jobs for freshers

The application processes the user's query and searches the available job data to return relevant opportunities.

---

# ✨ Features

- 🔎 Job search
- 🤖 AI-powered natural language queries
- 📥 Automated job data collection
- 🧹 Job data processing and cleaning
- 🗄️ Job data storage
- 🧠 LLM-powered query processing
- 🔗 LangChain integration
- 🔀 LangGraph-based workflow
- 💬 Interactive chatbot interface
- 📍 Location-based job search
- 🛠️ Skill-based job search

---

# 🏗️ System Architecture

The overall workflow of the application is:

```text
                  Job Sources
                      │
                      ▼
              Job Data Collection
                      │
                      ▼
              Data Processing
                      │
                      ▼
                Job Database
                      │
                      │
User Query ───────────┤
     │                │
     ▼                │
 Chat Interface       │
     │                │
     ▼                │
  Query Agent ────────┘
     │
     ▼
 Query Understanding
     │
     ▼
 Job Search / Filtering
     │
     ▼
 Relevant Job Results
     │
     ▼
    User
