# Apps Classification Using Text Mining Techniques
Using text mining techniques, this project classifies applications and games from the Windows Store based on their descriptions.

## Abstract
This project focuses on analyzing the top applications and games available on the Windows Store platform. By leveraging a dataset extracted from the Windows Store site, this project aims to uncover patterns from app descriptions. Through text mining techniques, especially classification and advanced analysis, this study seeks to understand whether text mining can help effectively classify the category or age ratings based on the descriptions from applications and games.

## Literature Review
Li et al. (2016) proposes classifying mobile applications by enriching app data with web knowledge. The research used the Vector Space Model (VSM), Naive Bayes (NB), and Latent Dirichlet Allocation (LDA) techniques and manually defined 7 main categories covering over 70% of the total applications. Given our dataset has 87 categories, including sub-categories, we focus on the main ones.

Olabenjo (2016) tackles app categorization using Naïve Bayes with metadata from top developer apps, showing Multinomial Naïve Bayes outperforms Bernoulli Naïve Bayes. Challenges arise in classifying gaming apps due to similar descriptions. The study emphasizes proper data collection, algorithm choice, and optimization.

While few studies use app descriptions for categorization, Johann et al. (2017) introduced SAFE, a method to match app features from descriptions (by developers) and reviews (by users). They use general textual patterns that are frequently used in app stores rather than large datasets or machine learning features and parameters.

Based on the key findings, while traditional approaches such as Naive Bayes and SVM remain relevant, the integration of advanced techniques like deep learning holds potential for enhancing accuracy and efficiency in app categorization tasks. As a result, this study have chosen to focus on Naïve Bayes and SVM, with a consideration for incorporating deep learning models. The evaluation will primarily focus on the F1 score, although we will also consider other metrics.

## Preprocessing

### Category Relabeling
With a total of 87 initial categories, we consolidated them into 15 main categories. Some categories with subcategories were merged into the main category for simplicity. Certain categories were reclassified, such as "Casino", "Puzzle & Trivia", "Strategy" were grouped as "Games"; "Music", "Photo & Video" were grouped as "Entertainment".

### Text Cleaning
All characters in the text were converted into lowercase. URLs and picture URLs were removed using regular expressions to eliminate irrelevant content. The text was filtered out any non-alphabetic characters, retaining only letters and apostrophes to maintain word integrity.

The text was also tokenized into individual words using the NLTK library. Stopwords were removed to focus on meaningful terms, and words with a length of two characters or less are filtered out, as they typically carry little semantic meaning. Redundant spaces, including leading and trailing spaces, were also stripped from the text to ensure proper formatting and readability.

## Feature Extraction
The dataset was split into training and testing sets, 80% of the data for training and 20% for testing. For feature extraction, we used CountVectorizer from scikit-learn to transform the textual descriptions of the apps into a numerical representation for machine learning algorithms. The CountVectorizer was applied to convert the text data into a matrix of token counts, where each row represents an app's description, and each column represents a unique word in the corpus.

## Evaluation and Results
Implemented Multinomial Naive Bayes (MNB), Support Vector Machine (SVM), and their tuned versions for app category prediction using text descriptions. Explored more on deep learning model, including Long Short-Term Memory (LSTM) for sequential pattern recognition and BERT, a transformer-based model, to capture contextual information and potentially improve classification performance.

| Model    | Accuracy | F1 Score | Precision | Recall |
|----------|----------|----------|-----------|--------|
| MNB      | 72.7%    | 68.9%    | 68.8%     | 72.7%  |
| MNB*     | 75.0%    | 74.3%    | 74.3%     | 75.0%  |
| SVM      | 71.7%    | 72.1%    | 73.0%     | 71.7%  |
| SVM*     | 73.6%    | 72.4%    | 74.9%     | 73.5%  |
| LSTM     | 67.6%    | 66.0%    | -         | -      |
| BERT     | 65.3%    | 57.4%    | -         | -      |

Model Performance for App Category. Note: An asterisk(*) indicates the fine-tuned model.


## Error Analysis
Generally, the default models shows relatively accurate classification for categories like "Games," "Entertainment," and "Personalization," but struggles with "Utilities & Tools" and "Productivity,”.

In the fine-tuned models, the misclassifications were reduced, but “Utilities & Tools” was still frequently misclassified as “Entertainment” and “Multimedia Design.” “Productivity” was often misclassified as “Utilities & Tools.” However, “Utilities & Tools” was not frequently misclassified as “Productivity,” indicating that it is difficult for the model to distinguish between these categories.

While the fine-tuned models show improvements by reducing misclassifications and increasing correct classifications, challenges remain. Distinguishing between overlapping categories like “Entertainment” and “Utilities & Tools” remains difficult, as does accurately categorizing “Productivity” apps. Further refinement of features or the incorporation of additional contextual information may be necessary to address these challenges and enhance the model's performance.
