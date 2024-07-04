import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import nltk
import re
import pickle

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load the dataset
df = pd.read_csv('UpdatedResumeDataSet.csv')

# Text cleaning function
def clean_resume(resume_text):
    resume_text = re.sub('http\S+\s*', ' ', resume_text)
    resume_text = re.sub('RT|cc', ' ', resume_text)
    resume_text = re.sub('#\S+', '', resume_text)
    resume_text = re.sub('@\S+', '  ', resume_text)
    resume_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resume_text)
    resume_text = re.sub(r'[^\x00-\x7f]', r' ', resume_text)
    resume_text = re.sub('\s+', ' ', resume_text)
    return resume_text

# Clean the resumes
df['cleaned_resume'] = df['Resume'].apply(clean_resume)

# Create TF-IDF vectorizer and transform the data
tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'), max_features=3000)
X = tfidf_vectorizer.fit_transform(df['cleaned_resume'])

# Encode the target labels
y = df['Category'].astype('category').cat.codes

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save the model and vectorizer
with open('clf.pkl', 'wb') as clf_file:
    pickle.dump(clf, clf_file)

with open('tfidf.pkl', 'wb') as tfidf_file:
    pickle.dump(tfidf_vectorizer, tfidf_file)

print("Model and vectorizer saved successfully.")


import streamlit as st
import pandas as pd
import pickle
import re
from PyPDF2 import PdfReader

# Load the trained model and vectorizer
with open('clf.pkl', 'rb') as clf_file:
    clf = pickle.load(clf_file)

with open('tfidf.pkl', 'rb') as tfidf_file:
    tfidf_vectorizer = pickle.load(tfidf_file)

# Text cleaning function
def clean_resume(resume_text):
    resume_text = re.sub('http\S+\s*', ' ', resume_text)
    resume_text = re.sub('RT|cc', ' ', resume_text)
    resume_text = re.sub('#\S+', '', resume_text)
    resume_text = re.sub('@\S+', '  ', resume_text)
    resume_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resume_text)
    resume_text = re.sub(r'[^\x00-\x7f]', r' ', resume_text)
    resume_text = re.sub('\s+', ' ', resume_text)
    return resume_text

# Category mapping (example, update based on your actual categories)
category_mapping = {
    15: "Java Developer",
    23: "Testing",
    8: "DevOps Engineer",
    20: "Python Developer",
    24: "Web Designing",
    12: "HR",
    13: "Hadoop",
    3: "Blockchain",
    10: "ETL Developer",
    18: "Operations Manager",
    6: "Data Science",
    22: "Sales",
    16: "Mechanical Engineer",
    1: "Arts",
    7: "Database",
    11: "Electrical Engineering",
    14: "Health and fitness",
    19: "PMO",
    4: "Business Analyst",
    9: "DotNet Developer",
    2: "Automation Testing",
    17: "Network Security Engineer",
    21: "SAP Developer",
    5: "Civil Engineer",
    0: "Advocate",
}

# Web app
def main():
    st.title("Resume Screening App")
    uploaded_file = st.file_uploader('Upload Resume', type=['txt', 'csv', 'pdf'])

    if uploaded_file is not None:
        if uploaded_file.type == 'text/csv':
            # Read the CSV file
            resume_df = pd.read_csv(uploaded_file)
            resume_text = ' '.join(resume_df.astype(str).values.flatten())
        elif uploaded_file.type == 'application/pdf':
            # Read the PDF file
            reader = PdfReader(uploaded_file)
            resume_text = ""
            for page in reader.pages:
                resume_text += page.extract_text()
        else:
            # Read the text file
            resume_text = uploaded_file.read().decode('utf-8')

        # Clean and process the resume
        cleaned_resume = clean_resume(resume_text)
        input_features = tfidf_vectorizer.transform([cleaned_resume])
        prediction_id = clf.predict(input_features)[0]

        # Get the category name from the mapping
        category_name = category_mapping.get(prediction_id, "Unknown")

        st.write("Predicted Category:", category_name)

if __name__ == "__main__":
    main()


