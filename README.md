# Resume MatchMate

## Overview
Resume MatchMate is a small Flask app that compares a resume (PDF) with a job description and gives a match percentage. It uses a Sentence-BERT model to turn both texts into embeddings and measures how similar they are.

## Features
- **Resume text extraction**: Reads the text from an uploaded PDF resume.
- **Semantic embeddings**: Splits each text into 500-character chunks, embeds them with a BERT-based Sentence Transformers model, and averages the chunk embeddings.
- **Match score**: The match rate is the cosine similarity between the resume and job description embeddings, shown as a percentage.

## Technologies Used
- Flask for web framework
- BERT from Sentence Transformers for NLP tasks
- PyPDF2 for PDF handling.

## Installation
To get started with Resume MatchMate:
1. Create and activate a python environment
2. Clone the repository: `git clone https://github.com/jeanemmanuelk/Resume-MatchMate.git`
3. Install dependencies: `pip install -r requirements.txt`
4. Run the application: `python app.py`

## Usage
Upload your resume in PDF format and paste the job description into the respective fields. Click "Find Match Rate" to see how well your resume aligns with the job description.

## Limitations
- The score is a rough text-similarity number, not a prediction of whether you would get the job. It does not check specific skills or requirements.
- Chunks are cut at fixed 500-character boundaries, which can split words and sentences.

## License
This project is licensed under the [MIT License](LICENSE).

## Contact
For any inquiries, please reach out to Jean-Emmanue Kouadio at emmanuelkj5@gmail.com .
