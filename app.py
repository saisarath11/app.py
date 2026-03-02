import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from transformers import pipeline
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
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

st.success(" ML Model Trained Successfully")


@st.cache_resource
def load_model():
    return pipeline(
        "text-generation",
        model="gpt2",
        device=-1
    )

generator = load_model()


name = st.text_input("Enter your name:")
email = st.text_input("Enter email:")
skills_input = st.text_area("Enter your skills:")
projects = st.text_area("Describe your project:")



if st.button("Generate Resume & Portfolio"):

    if not name or not email or not skills_input:
        st.warning(" Please fill all required fields.")
    else:

        # Predict Role
        skills_vector = vectorizer.transform([skills_input])
        predicted_role = model.predict(skills_vector)[0]

        st.subheader(" Predicted Job Role")
        st.success(predicted_role)
       

        objective_prompt = f"""
Career Objective:
A motivated {predicted_role} with strong skills in {skills_input},
"""

        bio_prompt = f"""
{name} is an aspiring {predicted_role} with expertise in {skills_input}.
"""

        project_prompt = f"""
This project titled '{projects}' focuses on
"""

        
        objective = generator(
            objective_prompt,
            max_new_tokens=60,
            temperature=0.6,
            do_sample=True,
            pad_token_id=50256
        )[0]["generated_text"]

        objective = objective.replace(objective_prompt, "").strip()

        
        bio = generator(
            bio_prompt,
            max_new_tokens=80,
            temperature=0.6,
            do_sample=True,
            pad_token_id=50256
        )[0]["generated_text"]

        bio = bio.replace(bio_prompt, "").strip()

        
        project_text = generator(
            project_prompt,
            max_new_tokens=100,
            temperature=0.6,
            do_sample=True,
            pad_token_id=50256
        )[0]["generated_text"]

        project_text = project_text.replace(project_prompt, "").strip()

   

        st.subheader(" AI Career Objective")
        st.write(objective)

        st.subheader(" AI Professional Bio")
        st.write(bio)

        st.subheader(" AI Project Description")
        st.write(project_text)

    

        resume_text = f"""
{name}
Email: {email}

Predicted Role: {predicted_role}

Career Objective:
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
{project_text}
"""

        st.subheader(" Generated Portfolio")
        st.text(portfolio_text)


        file_name = "AI_Resume_and_Portfolio.pdf"
        doc = SimpleDocTemplate(file_name, pagesize=A4)
        elements = []

        styles = getSampleStyleSheet()
        normal_style = styles["Normal"]

        full_text = resume_text + "\n\n" + portfolio_text

        for line in full_text.split("\n"):
            elements.append(Paragraph(line, normal_style))
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



