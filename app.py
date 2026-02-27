
!pip install -q scikit-learn transformers reportlab

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from transformers import pipeline
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

data = {
    "skills": [
        "python machine learning data analysis pandas",
        "html css javascript react",
        "aws cloud docker kubernetes",
        "network security ethical hacking cryptography",
        "deep learning neural networks artificial intelligence"
    ],
    "role": [
        "Data Scientist",
        "Web Developer",
        "Cloud Engineer",
        "Cyber Security Analyst",
        "AI Engineer"
    ]
}

df = pd.DataFrame(data)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["skills"])
y = df["role"]

model = MultinomialNB()
model.fit(X, y)

print("ML Model Trained Successfully!")

name = input("Enter your name: ")
email = input("Enter email: ")
skills_input = input("Enter your skills: ")
projects = input("Describe your project: ")

skills_vector = vectorizer.transform([skills_input])
predicted_role = model.predict(skills_vector)[0]

print("Predicted Job Role:", predicted_role)

generator = pipeline("text-generation", model="gpt2")

objective_prompt = f"Career Objective: A motivated {predicted_role} skilled in {skills_input} seeking"

objective = generator(
    objective_prompt,
    max_new_tokens=30,
    temperature=0.7,
    do_sample=True
)[0]["generated_text"]

print("\nAI Career Objective:\n", objective)

bio_prompt = f"Professional Bio: {name} is an aspiring {predicted_role} skilled in {skills_input}. "

bio = generator(
    bio_prompt,
    max_new_tokens=40,
    temperature=0.7,
    do_sample=True
)[0]["generated_text"]

print("\nAI Generated Bio:\n")
print(bio)

project_prompt = f"Project Description: This project involves {projects}. It focuses on"

project_text = generator(
    project_prompt,
    max_new_tokens=40,
    temperature=0.7,
    do_sample=True
)[0]["generated_text"]

print("\nAI Project Description:\n", project_text)

project_prompt = f"Portfolio Project Summary: This project involves {projects}. It focuses on"

project_summary = generator(
    project_prompt,
    max_new_tokens=50,
    temperature=0.7,
    do_sample=True
)[0]["generated_text"]

print("\nAI Enhanced Project Summary:\n")
print(project_summary)

resume_text = f"""
{name}

Email: {email}

Predicted Role: {predicted_role}

{objective}

Skills:
{skills_input}

{project_text}
"""

print(resume_text)

portfolio = f"""
===============================
        AI PORTFOLIO
===============================

Name: {name}
Email: {email}

Predicted Role: {predicted_role}


{bio}

{skills_input}

{project_summary}
"""
print(portfolio)

file_name = "AI_Resume.pdf"
c = canvas.Canvas(file_name, pagesize=A4)

y = 800
for line in resume_text.split("\n"):
    c.drawString(40, y, line)
    y -= 15
    if y < 40:
        c.showPage()
        y = 800

c.save()

from google.colab import files
files.download("AI_Resume.pdf")