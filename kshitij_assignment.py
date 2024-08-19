import pandas as pd

input_df = pd.read_excel('Input.xlsx')
urls = input_df['URL']
url_ids = input_df['URL_ID']

import requests
from bs4 import BeautifulSoup

def extract_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract the title
    title = soup.find('title').get_text()

    # Extract the main article text (this can vary by website)
    article_body = soup.find_all('p')
    article_text = " ".join([p.get_text() for p in article_body])

    return title, article_text

for url, url_id in zip(urls, url_ids):
    title, text = extract_text_from_url(url)

    # Save the extracted text to a file
    with open(f"{url_id}.txt", "w", encoding="utf-8") as file:
        file.write(f"{title}\n{text}")

import pandas as pd
import nltk
from textblob import TextBlob
import re

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('opinion_lexicon')

# Load positive and negative word lists (example)
positive_words = set(nltk.corpus.opinion_lexicon.words('positive-words.txt'))
negative_words = set(nltk.corpus.opinion_lexicon.words('negative-words.txt'))

def calculate_scores(text):
    words = nltk.word_tokenize(text.lower())

    # positive and negative scores
    positive_score = sum(1 for word in words if word in positive_words)
    negative_score = sum(1 for word in words if word in negative_words)

    # Polarity and subjectivity using TextBlob
    blob = TextBlob(text)
    polarity_score = blob.sentiment.polarity
    subjectivity_score = blob.sentiment.subjectivity

    # Sentence and word counts
    sentences = nltk.sent_tokenize(text)
    word_count = len(words)
    avg_sentence_length = sum(len(nltk.word_tokenize(sent)) for sent in sentences) / len(sentences)
    avg_word_length = sum(len(word) for word in words) / word_count

    # Complex words calculation
    complex_words = [word for word in words if len(nltk.word_tokenize(word)) > 2]
    percentage_of_complex_words = len(complex_words) / word_count * 100

    # FOG index
    fog_index = 0.4 * (avg_sentence_length + percentage_of_complex_words)

    # Personal pronouns count
    personal_pronouns = len(re.findall(r'\b(?:I|we|my|ours|us)\b', text, re.IGNORECASE))

    # Syllable count per word
    syllable_per_word = sum([len(re.findall(r'[aeiouy]', word)) for word in words]) / word_count

    return {
        "positive_score": positive_score,
        "negative_score": negative_score,
        "polarity_score": polarity_score,
        "subjectivity_score": subjectivity_score,
        "avg_sentence_length": avg_sentence_length,
        "percentage_of_complex_words": percentage_of_complex_words,
        "fog_index": fog_index,
        "complex_word_count": len(complex_words),
        "word_count": word_count,
        "syllable_per_word": syllable_per_word,
        "personal_pronouns": personal_pronouns,
        "avg_word_length": avg_word_length,
    }

results = []
for url_id, url in zip(url_ids, urls):
    with open(f"{url_id}.txt", "r", encoding="utf-8") as file:
        text = file.read()

    scores = calculate_scores(text)
    results.append([url_id, url] + list(scores.values()))

output_df = pd.DataFrame(results, columns=[
    'URL_ID', 'URL', 'POSITIVE SCORE', 'NEGATIVE SCORE', 'POLARITY SCORE',
    'SUBJECTIVITY SCORE', 'AVG SENTENCE LENGTH', 'PERCENTAGE OF COMPLEX WORDS',
    'FOG INDEX', 'COMPLEX WORD COUNT', 'WORD COUNT', 'SYLLABLE PER WORD',
    'PERSONAL PRONOUNS', 'AVG WORD LENGTH'
])

output_df.to_excel("Output Data Structure.xlsx", index=False)