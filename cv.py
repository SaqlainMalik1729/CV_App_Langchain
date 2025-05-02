import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os
import tempfile
from langchain.docstore.document import Document
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

# Pydantic model for resume data
class ResumeData(BaseModel):
    name: Optional[str] = Field(description="Full name of the individual", default=None)
    contact_details: Optional[dict] = Field(description="Contact information (e.g., email, phone, address)", default=None)
    key_skills: List[str] = Field(description="List of key skills or competencies", default=[])
    projects: List[dict] = Field(description="List of projects with title, description, and technologies", default=[])
    summary: Optional[str] = Field(description="Overall summary of the individual's qualifications", default=None)

# Load environment variables
load_dotenv()

# Streamlit page configuration
st.set_page_config(page_title="Resume-Job Similarity", layout="wide")

# Inject Tailwind CSS and Heroicons via CDN
st.markdown("""
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; }
        .stButton>button {
            background-color: #3b82f6;
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 0.375rem;
            border: none;
            transition: background-color 0.3s;
        }
        .stButton>button:hover {
            background-color: #2563eb;
        }
        .card { transition: transform 0.2s; }
        .card:hover { transform: translateY(-4px); }
        .progress-bar { background-color: #e5e7eb; border-radius: 0.375rem; overflow: hidden; }
        .progress-fill { height: 1.5rem; transition: width 0.5s ease-in-out; }
        .icon { width: 1.5rem; height: 1.5rem; vertical-align: middle; margin-right: 0.5rem; }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="bg-gradient-to-r from-blue-600 to-indigo-600 text-white py-6 px-4 text-center rounded-lg shadow-lg">
        <h1 class="text-4xl font-bold">Resume & Job Description Similarity Checker</h1>
        <p class="mt-2 text-lg">Upload a resume and enter a job description to find the perfect match!</p>
    </div>
""", unsafe_allow_html=True)

# Initialize LangChain components
@st.cache_resource
def initialize_langchain():
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    emb_model = OpenAIEmbeddings()
    model = ChatOpenAI()
    return text_splitter, emb_model, model

text_splitter, emb_model, model = initialize_langchain()

# Initialize resume parser
parser = PydanticOutputParser(pydantic_object=ResumeData)
prompt = PromptTemplate(
    template"""You are an expert resume parser. Extract the following information from the provided resume text:
        - Full name
        - Contact details (e.g., email, phone, address)
        - Key skills (list of skills or competencies)
        - Projects (title, description, and technologies used for each)
        - Overall summary (a brief summary of the individual's qualifications and experience)

        Resume text:
        {resume_docs}
        
        Provide the result in the given JSON format:
        {format_instruction}
    """,
    input_variables=['resume_docs'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)
chain = prompt | model | parser

# Input Section
st.markdown('<div class="mt-8">', unsafe_allow_html=True)
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<h2 class="text-xl font-semibold text-gray-700">Upload Resume</h2>', unsafe_allow_html=True)
    resume_file = st.file_uploader("", type=["pdf"], key="resume", help="Upload a PDF resume")

with col2:
    st.markdown('<h2 class="text-xl font-semibold text-gray-700">Enter Job Description</h2>', unsafe_allow_html=True)
    job_description = st.text_area("", height=150, key="job_desc", help="Paste or type the job description")

# Compute Button
st.markdown('<div class="text-center mt-6">', unsafe_allow_html=True)
if st.button("Process", key="compute"):
    if resume_file is None:
        st.markdown('<p class="text-red-500 text-center">Please upload a resume PDF.</p>', unsafe_allow_html=True)
    elif not job_description.strip():
        st.markdown('<p class="text-red-500 text-center">Please enter a job description.</p>', unsafe_allow_html=True)
    else:
        with st.spinner("Analyzing..."):
            try:
                # Save uploaded PDF to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(resume_file.read())
                    tmp_file_path = tmp_file.name

                # Load and process resume
                resume_loader = PyPDFLoader(tmp_file_path)
                resume_docs = resume_loader.load()
                resume_chunks = text_splitter.split_documents(resume_docs)

                # Parse resume details
                resume_text = " ".join([doc.page_content for doc in resume_docs])
                resume_detail = chain.invoke({'resume_docs': resume_text})

                # Create a Document from job description
                job_doc = Document(page_content=job_description)
                job_chunks = text_splitter.split_documents([job_doc])

                # Create FAISS vector store for resume chunks
                resume_vector_store = FAISS.from_documents(resume_chunks, emb_model)

                # Combine job description chunks into a single query
                job_description_text = " ".join([chunk.page_content for chunk in job_chunks])

                # Perform similarity search
                k = 10  # Number of top matches to retrieve
                search_results = resume_vector_store.similarity_search_with_score(query=job_description_text, k=k)

                # Display Resume Details
                st.markdown('<h2 class="text-2xl font-bold text-gray-800 mt-8">Resume Details</h2>', unsafe_allow_html=True)
                st.markdown('<div class="card bg-white p-6 rounded-lg shadow-md border border-gray-200 mb-6">', unsafe_allow_html=True)

                # Name
                st.markdown("""
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                            <p class="text-lg font-semibold text-gray-700">
                                <svg class="icon" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"></path></svg>
                                Name
                            </p>
                            <p class="text-gray-600">{}</p>
                        </div>
                """.format(resume_detail.name if resume_detail.name else "Not provided"), unsafe_allow_html=True)

                # Contact Details
                contact_html = "<div><p class='text-lg font-semibold text-gray-700'><svg class='icon' fill='none' stroke='currentColor' viewBox='0 0 24 24' xmlns='http://www.w3.org/2000/svg'><path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z'></path></svg>Contact Details</p>"
                if resume_detail.contact_details:
                    for key, value in resume_detail.contact_details.items():
                        contact_html += f"<p class='text-gray-600'>{key.capitalize()}: {value}</p>"
                else:
                    contact_html += "<p class='text-gray-600'>Not provided</p>"
                contact_html += "</div>"
                st.markdown(contact_html, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

                # Key Skills
                st.markdown("""
                    <p class="text-lg font-semibold text-gray-700 mt-4">
                        <svg class="icon" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"></path></svg>
                        Key Skills
                    </p>
                """, unsafe_allow_html=True)
                if resume_detail.key_skills:
                    skills_html = "<div class='flex flex-wrap gap-2'>"
                    for skill in resume_detail.key_skills:
                        skills_html += f"<span class='bg-blue-100 text-blue-800 text-sm font-medium px-2.5 py-0.5 rounded'>{skill}</span>"
                    skills_html += "</div>"
                    st.markdown(skills_html, unsafe_allow_html=True)
                else:
                    st.markdown("<p class='text-gray-600'>No skills provided</p>", unsafe_allow_html=True)

                # Projects
                st.markdown("""
                    <p class="text-lg font-semibold text-gray-700 mt-4">
                        <svg class="icon" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                        Projects
                    </p>
                """, unsafe_allow_html=True)
                if resume_detail.projects:
                    for project in resume_detail.projects:
                        title = project.get('title', 'Untitled')
                        description = project.get('description', 'No description')
                        technologies = project.get('technologies', 'Not specified')
                        st.markdown(f"""
                            <div class="mt-2">
                                <p class="text-gray-700 font-medium">{title}</p>
                                <p class="text-gray-600">{description}</p>
                                <p class="text-gray-500 text-sm">Technologies: {technologies}</p>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("<p class='text-gray-600'>No projects provided</p>", unsafe_allow_html=True)

                # Summary
                st.markdown("""
                    <p class="text-lg font-semibold text-gray-700 mt-4">
                        <svg class="icon" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path></svg>
                        Summary
                    </p>
                """, unsafe_allow_html=True)
                st.markdown(f"<p class='text-gray-600'>{resume_detail.summary if resume_detail.summary else 'No summary provided'}</p>", unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

                # Results Section
                st.markdown('<h2 class="text-2xl font-bold text-gray-800 mt-8">Results</h2>', unsafe_allow_html=True)
                st.markdown('<p class="text-gray-600 mb-4">Top matching resume sections and their similarity scores:</p>', unsafe_allow_html=True)
                total_score = 0
                num_results = len(search_results)
                for i, (doc, score) in enumerate(search_results, 1):
                    similarity = 1 - score
                    total_score += similarity

                # Compute average similarity and convert to percentage
                average_similarity = total_score / num_results if num_results > 0 else 0
                average_similarity_percent = int(average_similarity * 100)

                # Display average similarity with progress bar
                st.markdown(f"""
                    <h3 class="text-xl font-semibold text-gray-700 mt-6">Average Similarity</h3>
                    <p class="text-3xl font-bold text-blue-600">{average_similarity_percent}%</p>
                    <div class="progress-bar mt-2">
                        <div class="progress-fill bg-blue-500" style="width: {average_similarity_percent}%"></div>
                    </div>
                """, unsafe_allow_html=True)

                # Determine recommendation
                if average_similarity_percent <= 40:
                    recommendation = "Not Applicable"
                    color = "text-red-500"
                elif 40 < average_similarity_percent <= 55:
                    recommendation = "Good"
                    color = "text-yellow-500"
                elif 55 < average_similarity_percent <= 70:
                    recommendation = "Recommended"
                    color = "text-green-500"
                else:
                    recommendation = "Highly Recommended"
                    color = "text-blue-500"

                # Display recommendation
                st.markdown(f"""
                    <h3 class="text-xl font-semibold text-gray-700 mt-4">Recommendation</h3>
                    <p class="{color} text-2xl font-bold">{recommendation}</p>
                """, unsafe_allow_html=True)

                # Clean up temporary file
                os.unlink(tmp_file_path)

            except Exception as e:
                st.markdown(f'<p class="text-red-500 text-center">An error occurred: {str(e)}</p>', unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="bg-gray-100 py-4 px-4 text-center text-gray-600 mt-8 rounded-lg">
        <p>Powered by <span class="font-semibold">LangChain</span> and <span class="font-semibold">Streamlit</span></p>
        <p class="text-sm mt-1">© 2025 Resume Similarity Checker</p>
    </div>
""", unsafe_allow_html=True)
