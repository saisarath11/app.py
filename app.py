import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import pipeline

st.set_page_config(
    page_title="AI Resume & Portfolio Builder",
    layout="wide"
)

st.markdown("""
<style>
.main { background-color: #0e1117; }
h1, h2, h3 { color: #4CAF50; }
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 8px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

st.title(" AI Resume & Portfolio Builder")


data = [
    ("python machine learning data analysis pandas numpy", "Data Scientist"),
    ("html css javascript react ui ux", "Frontend Developer"),
    ("java spring boot api backend database", "Backend Developer"),
    ("c c++ embedded systems microcontroller", "Embedded Engineer")
]

texts = [x[0] for x in data]
labels = [x[1] for x in data]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

ml_model = LogisticRegression()
ml_model.fit(X, labels)


@st.cache_resource
def load_model():
    return pipeline(
        task="text2text-generation",
        model="google/flan-t5-base",
        device=-1
    )

generator = load_model()

def generate_text(prompt):
    response = generator(
        prompt,
        max_new_tokens=120,
        do_sample=True,
        temperature=0.6,
        top_p=0.85,
        repetition_penalty=1.3
    )
    return response[0]["generated_text"].strip()

col1, col2 = st.columns(2)

with col1:
    name = st.text_input("Full Name")
    email = st.text_input("Email")
    skills_input = st.text_area("Enter Your Skills (example: python, c++, machine learning)")

with col2:
    project_title = st.text_input("Project Title")
    project_desc = st.text_area("Project Description")


if st.button("Generate Resume & Portfolio"):

    if skills_input.strip() == "":
        st.warning("Please enter your skills.")
        st.stop()

    if project_desc.strip() == "":
        project_desc = "AI Resume & Portfolio Builder that generates professional resumes and portfolios based on user skills using machine learning."

  
    skills_vector = vectorizer.transform([skills_input])
    predicted_role = ml_model.predict(skills_vector)[0]

    st.success(f" Predicted Job Role: {predicted_role}")

  

    objective_prompt = f"""
Write a professional career objective for a college student.

Role: {predicted_role}
Skills: {skills_input}

Requirements:
- Write 2 to 3 sentences
- Sound confident and professional
- Mention learning and technical skills
"""

    bio_prompt = f"""
Write a short professional bio in third person.

Name: {name}
Role: aspiring {predicted_role}
Skills: {skills_input}

Requirements:
- 2 to 3 sentences
- Describe the student as motivated and passionate about technology
"""

    project_prompt = f"""
Write a professional project description.

Project Name: {project_title}

Details:
{project_desc}

Requirements:
- 3 sentences
- Explain the purpose of the project
- Mention AI or programming technologies
- Explain how it helps users
"""

    portfolio_prompt = f"""
Write a professional portfolio summary.

Name: {name}
Role: aspiring {predicted_role}
Skills: {skills_input}
Project: {project_title}

Requirements:
- 3 sentences
- Describe technical skills and projects
- Sound professional
"""

    objective = generate_text(objective_prompt)
    bio = generate_text(bio_prompt)
    project_text = generate_text(project_prompt)
    portfolio_summary = generate_text(portfolio_prompt)

   
    st.markdown(" Career Objective")
    st.info(objective)

    st.markdown(" Professional Bio")
    st.success(bio)

    st.markdown(" Project Description")
    st.warning(project_text)

    st.markdown(" Portfolio Summary")
    st.success(portfolio_summary)

    
    resume_text = f"""
{name}
Email: {email}

Predicted Role:
{predicted_role}

Career Objective:
{objective}

Professional Bio:
{bio}

Skills:
{skills_input}

Project:
{project_text}
"""

    portfolio_text = f"""
{name}

Aspiring Role:
{predicted_role}

Professional Bio:
{bio}

Skills:
{skills_input}

Project:
{project_title}

Project Summary:
{project_text}

Portfolio Summary:
{portfolio_summary}
"""

    st.markdown(" Generated Resume")
    st.text_area("Resume Preview", resume_text, height=300)

    st.markdown(" Generated Portfolio")
    st.text_area("Portfolio Preview", portfolio_text, height=300)

    st.download_button(
        label="⬇ Download Resume",
        data=resume_text,
        file_name="resume.txt",
        mime="text/plain"
    )

    st.download_button(
        label="⬇ Download Portfolio",
        data=portfolio_text,
        file_name="portfolio.txt",
        mime="text/plain"
    )
















