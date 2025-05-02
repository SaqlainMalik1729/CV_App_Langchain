# CV_App_Langchain
# Resume & Job Description Similarity Checker

A Streamlit-based web application that analyzes the similarity between a resume (PDF) and a job description using LangChain, OpenAI embeddings, and FAISS vector search. Extracts structured resume data and provides a similarity score with a recommendation.

## Features
- Upload a PDF resume and input a job description.
- Extracts resume details (name, contact, skills, projects, summary) using a Pydantic model and LangChain.
- Computes similarity between resume sections and job description using FAISS and OpenAI embeddings.
- Displays average similarity as a percentage with a progress bar.
- Provides a recommendation based on similarity score (Not Applicable, Good, Recommended, Highly Recommended).
- Responsive UI with Tailwind CSS and Heroicons.

## Requirements
- Python 3.8+
- Libraries: `streamlit`, `langchain`, `langchain-openai`, `faiss-cpu`, `PyPDF2`, `python-dotenv`, `pydantic`
- OpenAI API key (stored in `.env`)

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your OpenAI API key:
   ```env
   OPENAI_API_KEY=your-api-key
   ```
4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Open the app in your browser (default: `http://localhost:8501`).
2. Upload a PDF resume and enter a job description.
3. Click "Process" to analyze.
4. View extracted resume details, similarity score, and recommendation.

## Code Structure
- **Pydantic Model**: Defines `ResumeData` for structured resume parsing.
- **LangChain Components**: Uses `ChatOpenAI`, `OpenAIEmbeddings`, `RecursiveCharacterTextSplitter`, and `PromptTemplate` for resume parsing and similarity analysis.
- **FAISS**: Vector store for similarity search.
- **Streamlit**: Handles UI, file upload, and result display.
- **Temp File Management**: Processes PDF resumes securely.

## Limitations
- Supports PDF resumes only.
- Requires a valid OpenAI API key.
- Similarity accuracy depends on resume and job description quality.

## License
MIT License

## Acknowledgments
- Powered by [LangChain](https://langchain.com) and [Streamlit](https://streamlit.io).
- Styling with [Tailwind CSS](https://tailwindcss.com) and [Heroicons](https://heroicons.com).
