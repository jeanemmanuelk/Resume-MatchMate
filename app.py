from flask import Flask, jsonify, request, render_template
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
import numpy as np

app = Flask(__name__)

# Load the BERT model
model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')

def extract_text_from_pdf(pdf_file):
    try:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() if page.extract_text() else ""
        return text
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""

def split_into_segments(text, max_length=500):
    # Split the text into segments of max_length 500 since the bert model as max of 512 tokens
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

def calculate_embedding_for_text(text):
    segments = split_into_segments(text)
    embeddings = [model.encode(segment, convert_to_tensor=True) for segment in segments]
    # Average the embeddings
    avg_embedding = np.mean(embeddings, axis=0)
    return avg_embedding

def calculate_match_rate(resume_text, job_description):
    try:
        # Calculate embeddings for resume and job description
        resume_embedding = calculate_embedding_for_text(resume_text)
        job_desc_embedding = calculate_embedding_for_text(job_description)
        
        # Compute cosine similarity
        cosine_similarity = util.pytorch_cos_sim(resume_embedding, job_desc_embedding)
        
        # Convert to a percentage for the match rate
        match_rate = cosine_similarity.item() * 100
        return round(match_rate, 2)
    except Exception as e:
        print(f"Error calculating match rate: {e}")
        return 0

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Assume the PDF file is sent under the name 'resume'
        resume_file = request.files['resume']
        
        # Ensure a file is received and it's a PDF
        if resume_file and allowed_file(resume_file.filename):
            # Extract text from the resume PDF
            resume_text = extract_text_from_pdf(resume_file)
            
            # Get job description text from the form
            job_description = request.form['job_description']
            
            # Calculate the semantic match rate using BERT
            match_rate = calculate_match_rate(resume_text, job_description)
            
            # Return the match rate as a JSON response
            return jsonify({
                'resume_text': resume_text[:500],  # Returning first 500 chars for brevity
                'job_description': job_description[:500],  # Same as above
                'match_rate': f"{match_rate}%"
            })
    
    # Render the main page with the form
    return render_template('index.html')

# @app.route('/feedback', methods=['POST'])
# def feedback():
#     feedback_text = request.form['feedback']
#     # Here you can process the feedback, save it to a file or database, etc.
#     print("Received Feedback:", feedback_text)
#     return jsonify({'status': 'success'})

def allowed_file(filename):
    # Enhanced file validation
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'pdf', 'other', 'file', 'extensions'}

if __name__ == '__main__':
    app.run(debug=True)