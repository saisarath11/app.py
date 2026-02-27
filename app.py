import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from transformers import pipeline
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import os

st.set_page_config(page_title=" AI Resume & Portfolio Builder", layout="centered")

st.title(" AI Resume & Portfolio Builder")


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

st.success("ML Model Trained Successfully!")


@st.cache_resource
def load_model():
    return pipeline("text-generation", model="sshleifer/tiny-gpt2")

generator = load_model()


name = st.text_input("Enter your name:")
email = st.text_input("Enter email:")
skills_input = st.text_area("Enter your skills:")
projects = st.text_area("Describe your project:")


if st.button("Generate Resume & Portfolio"):

    if not name or not email or not skills_input:
        st.warning("Please fill all required fields.")
    else:

        skills_vector = vectorizer.transform([skills_input])
        predicted_role = model.predict(skills_vector)[0]

        st.subheader(" Predicted Job Role")
        st.success(predicted_role)

       
        objective_prompt = f"Career Objective: A motivated {predicted_role} skilled in {skills_input} seeking"

        objective = generator(
            objective_prompt,
            max_new_tokens=40,
            temperature=0.7,
            do_sample=True
        )[0]["generated_text"]

        st.subheader(" AI Career Objective")
        st.write(objective)

        bio_prompt = f"{name} is an aspiring {predicted_role} skilled in {skills_input}. "

        bio = generator(
            bio_prompt,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True
        )[0]["generated_text"]

        st.subheader("👤 AI Generated Bio")
        st.write(bio)

       
        project_prompt = f"Project Description: This project involves {projects}. It focuses on"

        project_text = generator(
            project_prompt,
            max_new_tokens=60,
            temperature=0.7,
            do_sample=True
        )[0]["generated_text"]

        st.subheader(" AI Project Description")
        st.write(project_text)

      
        project_prompt2 = f"Portfolio Project Summary: This project involves {projects}. It focuses on"

        project_summary = generator(
            project_prompt2,
            max_new_tokens=60,
            temperature=0.7,
            do_sample=True
        )[0]["generated_text"]

        st.subheader(" AI Enhanced Project Summary")
        st.write(project_summary)


        resume_text = f"""
{name}
Email: {email}

Predicted Role: {predicted_role}

{objective}

Skills:
{skills_input}

Project Description:
{project_text}
"""

        st.subheader(" Generated Resume")
        st.text(resume_text)

      

        portfolio_text = f"""


Name: {name}
Email: {email}

Predicted Role: {predicted_role}

Professional Bio:
{bio}

Skills:
{skills_input}

Project Summary:
{project_summary}
"""

        st.subheader(" Generated Portfolio")
        st.text(portfolio_text)


        full_text = resume_text + "\n\n" + portfolio_text

        file_name = "AI_Resume_and_Portfolio.pdf"
        c = canvas.Canvas(file_name, pagesize=A4)

        y = 800
        for line in full_text.split("\n"):
            c.drawString(40, y, line)
            y -= 15
            if y < 40:
                c.showPage()
                y = 800

        c.save()

        with open(file_name, "rb") as f:
            st.download_button(
                " Download Resume & Portfolio PDF",
                f,
                file_name="AI_Resume_and_Portfolio.pdf",
                mime="application/pdf"
            )

        os.remove(file_name)

