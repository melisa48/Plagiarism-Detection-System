# Plagiarism Detection System
This Python-based plagiarism detection system compares multiple documents using TF-IDF and semantic analysis techniques to identify similarities and potential instances of plagiarism.

## Features
- Compares multiple documents for similarities
- Uses both TF-IDF and semantic analysis methods
- Flags potential plagiarism based on a customizable threshold
- Generates detailed similarity reports
- Produces similarity heatmaps for visual analysis
- Calculates document uniqueness scores
- Exports results to CSV files for further analysis

## Requirements
- Python 3.x
- NLTK
- scikit-learn
- matplotlib
- seaborn
- pandas
- sentence-transformers

## Installation
1. Clone the repository or download the script.
2. Install the required packages:
- `pip install nltk scikit-learn matplotlib seaborn pandas sentence-transformers`
3. Download required NLTK data:
- `import nltk`
- `nltk.download('stopwords')`
- `nltk.download('punkt')`
-` nltk.download('wordnet')`

## Usage
Add your documents to the `documents` list in the script.
Set the `plagiarism_threshold` as desired (default is 70%).
Run the script:
- `python plagiarism_detection.py`

## Output
- Console output with similarity scores and potential plagiarism flags
- TF-IDF and semantic similarity reports (saved as text files)
- Similarity heatmaps (displayed and can be saved)
- CSV files with detailed similarity results
- Document uniqueness scores

## Customization
- Adjust the plagiarism_threshold to change sensitivity
- Modify the preprocess_text function to customize text preprocessing
- Change the semantic model in SentenceTransformer() for different embedding results
## Note
- This system is for educational purposes and should not be solely relied upon for detecting plagiarism in academic or professional settings.

## Contributing
- Contributions to improve Plagiarism-Detection-System are welcome. Please follow the standard fork-and-pull request workflow.
