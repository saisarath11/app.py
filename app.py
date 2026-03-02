import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
import os

st.set_page_config(page_title="AI Resume & Portfolio Builder", layout="centered")

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

st.success("ML Model Trained Successfully")


name = st.text_input("Enter your name:")
email = st.text_input("Enter email:")
skills_input = st.text_area("Enter your skills:")
project_name = st.text_input("Enter your project title:")
project_desc = st.text_area("Describe your project briefly:")


if st.button("Generate Resume & Portfolio"):

    if not name or not email or not skills_input:
        st.warning(" Please fill all required fields.")
    else:

        skills_vector = vectorizer.transform([skills_input])
        predicted_role = model.predict(skills_vector)[0]

        st.subheader(" Predicted Job Role")
        st.success(predicted_role)


        objective = f"""
Motivated and detail-oriented {predicted_role} with strong knowledge in {skills_input}. 
Passionate about solving real-world problems using technology and continuously improving technical expertise.
"""

        bio = f"""
{name} is an aspiring {predicted_role} with a solid foundation in {skills_input}. 
Demonstrates strong analytical thinking, problem-solving skills, and dedication to delivering high-quality solutions.
"""

        project_text = f"""
{project_name} is a practical implementation project where {project_desc}. 
The project highlights technical proficiency in {skills_input} and demonstrates the ability to design and build real-world applications.
"""


        st.subheader(" Career Objective")
        st.write(objective)

        st.subheader(" Professional Bio")
        st.write(bio)

        st.subheader(" Project Description")
        st.write(project_text)


        resume_text = f"""
{name}
Email: {email}

Predicted Role: {predicted_role}

Career Objective:
{objective}

Skills:
{skills_input}

Project:
{project_text}
"""

        st.subheader(" Generated Resume")
        st.text(resume_text)


        portfolio_text = f"""
Name: {name}
Email: {email}

Role: {predicted_role}

Professional Bio:
{bio}

Skills:
{skills_input}

Project Summary:
{project_text}
"""

        st.subheader(" Generated Portfolio")
        st.text(portfolio_text)


        file_name = "AI_Resume_and_Portfolio.pdf"
        doc = SimpleDocTemplate(file_name, pagesize=A4)
        elements = []
        styles = getSampleStyleSheet()
        style = styles["Normal"]

        full_text = resume_text + "\n\n" + portfolio_text

        for line in full_text.split("\n"):
            elements.append(Paragraph(line, style))
            elements.append(Spacer(1, 8))

        doc.build(elements)

        with open(file_name, "rb") as f:
            st.download_button(
                "⬇ Download Resume & Portfolio PDF",
                f,
                file_name=file_name,
                mime="application/pdf"
            )

        os.remove(file_name)




