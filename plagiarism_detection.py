import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Download necessary NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize stopwords, lemmatizer, and sentence transformer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def preprocess_text(text, remove_stopwords=True, lemmatize=True):
    """Preprocess the text: lowercase, remove punctuation, tokenize, optionally remove stopwords and lemmatize."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = word_tokenize(text)
    if remove_stopwords:
        words = [word for word in words if word not in stop_words]
    if lemmatize:
        words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

def calculate_similarity(doc1, doc2, method='tfidf'):
    """Calculate similarity between two documents using TF-IDF or semantic embeddings."""
    if method == 'tfidf':
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([doc1, doc2])
        return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    elif method == 'semantic':
        embeddings = sentence_model.encode([doc1, doc2])
        return cosine_similarity(embeddings)[0][1]

def compare_multiple_documents(documents, remove_stopwords=True, lemmatize=True, method='tfidf'):
    """Compare multiple documents and calculate pairwise similarities."""
    processed_docs = [preprocess_text(doc, remove_stopwords, lemmatize) for doc in documents]
    results = []
    for (i, doc1), (j, doc2) in combinations(enumerate(processed_docs, 1), 2):
        similarity = calculate_similarity(doc1, doc2, method)
        results.append({
            "doc1_id": i,
            "doc2_id": j,
            "similarity": similarity * 100,  # Convert to percentage
            "doc1_processed": doc1,
            "doc2_processed": doc2
        })
    return results

def plot_similarity_heatmap(comparison_results, num_documents):
    """Plot a heatmap of document similarities."""
    similarity_matrix = [[0 for _ in range(num_documents)] for _ in range(num_documents)]
    for result in comparison_results:
        i, j = result['doc1_id'] - 1, result['doc2_id'] - 1
        similarity_matrix[i][j] = similarity_matrix[j][i] = result['similarity']
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_matrix, annot=True, cmap="YlOrRd", vmin=0, vmax=100)
    plt.title("Document Similarity Heatmap")
    plt.xlabel("Document ID")
    plt.ylabel("Document ID")
    plt.show()

def generate_similarity_report(comparison_results, documents, plagiarism_threshold):
    """Generate a detailed similarity report with plagiarism flagging."""
    report = "Detailed Similarity Report\n"
    report += "==========================\n\n"
    
    for result in comparison_results:
        report += f"Document {result['doc1_id']} vs Document {result['doc2_id']}:\n"
        report += f"Similarity: {result['similarity']:.2f}%\n"
        if result['similarity'] > plagiarism_threshold:
            report += "*** POTENTIAL PLAGIARISM DETECTED ***\n"
        report += f"Doc {result['doc1_id']} (original): {documents[result['doc1_id']-1]}\n"
        report += f"Doc {result['doc2_id']} (original): {documents[result['doc2_id']-1]}\n"
        report += f"Doc {result['doc1_id']} (processed): {result['doc1_processed']}\n"
        report += f"Doc {result['doc2_id']} (processed): {result['doc2_processed']}\n\n"
    
    return report

def analyze_document_uniqueness(comparison_results, num_documents):
    """Analyze the uniqueness of each document based on its similarities."""
    uniqueness_scores = [100] * num_documents
    for result in comparison_results:
        i, j = result['doc1_id'] - 1, result['doc2_id'] - 1
        similarity = result['similarity']
        uniqueness_scores[i] -= similarity / (num_documents - 1)
        uniqueness_scores[j] -= similarity / (num_documents - 1)
    return [max(0, score) for score in uniqueness_scores]

if __name__ == "__main__":
    documents = [
        "Artificial intelligence is the future of technology.",
        "The future of technology lies in artificial intelligence.",
        "Machine learning, a subset of AI, is revolutionizing various industries.",
        "Natural language processing is an important field within artificial intelligence.",
        "The impact of AI on future technology cannot be overstated.",
        "Deep learning algorithms are transforming the field of computer vision.",
        "Robotics and AI are combining to create autonomous systems.",
        "Big data analytics relies heavily on machine learning techniques.",
        "The ethical implications of AI are a topic of ongoing debate.",
        "Quantum computing may revolutionize the capabilities of AI systems."
    ]

    plagiarism_threshold = 70.0  # Set threshold for flagging potential plagiarism

    print("Comparing documents using TF-IDF:")
    tfidf_results = compare_multiple_documents(documents, method='tfidf')
    
    print("\nComparing documents using Semantic Analysis:")
    semantic_results = compare_multiple_documents(documents, method='semantic')

    for method, results in [("TF-IDF", tfidf_results), ("Semantic", semantic_results)]:
        print(f"\n{method} Analysis Results:")
        for result in results:
            print(f"\nDocument {result['doc1_id']} vs Document {result['doc2_id']}:")
            print(f"Similarity: {result['similarity']:.2f}%")
            if result['similarity'] > plagiarism_threshold:
                print("*** POTENTIAL PLAGIARISM DETECTED ***")

        most_similar = max(results, key=lambda x: x['similarity'])
        print(f"\nMost similar pair: Document {most_similar['doc1_id']} and Document {most_similar['doc2_id']}")
        print(f"Similarity: {most_similar['similarity']:.2f}%")

        least_similar = min(results, key=lambda x: x['similarity'])
        print(f"\nLeast similar pair: Document {least_similar['doc1_id']} and Document {least_similar['doc2_id']}")
        print(f"Similarity: {least_similar['similarity']:.2f}%")

        # Generate and save detailed report
        report = generate_similarity_report(results, documents, plagiarism_threshold)
        with open(f"{method.lower()}_similarity_report.txt", "w") as f:
            f.write(report)
        print(f"\nDetailed {method} similarity report saved to '{method.lower()}_similarity_report.txt'")

        # Plot similarity heatmap
        plot_similarity_heatmap(results, len(documents))

        # Analyze document uniqueness
        uniqueness_scores = analyze_document_uniqueness(results, len(documents))
        print(f"\n{method} Document Uniqueness Scores:")
        for i, score in enumerate(uniqueness_scores, 1):
            print(f"Document {i}: {score:.2f}%")

        # Export results to CSV
        df = pd.DataFrame(results)
        df.to_csv(f"{method.lower()}_similarity_results.csv", index=False)
        print(f"\n{method} similarity results exported to '{method.lower()}_similarity_results.csv'")

print("\nAnalysis complete.")
