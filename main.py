import nltk
import ssl
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
from snownlp import SnowNLP
import jieba

ssl._create_default_https_context = ssl._create_unverified_context

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

with open('mobydick.txt', 'r', encoding='utf-8') as file:
    moby_dick_text = file.read()

s = SnowNLP(moby_dick_text)

tokens = nltk.word_tokenize(moby_dick_text)

stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

pos_tags = nltk.pos_tag(filtered_tokens)

pos_counts = FreqDist(tag for word, tag in pos_tags)
top_pos = pos_counts.most_common(5)
print("Top 5 Parts of Speech and their frequencies:")
for tag, count in top_pos:
    print(f"{tag}: {count}")

lemmatizer = WordNetLemmatizer()

def wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif treebank_tag.startswith('N'):
        return nltk.corpus.wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return nltk.corpus.wordnet.ADV
    else:
        return nltk.corpus.wordnet.NOUN

lemmas = [lemmatizer.lemmatize(word, pos=wordnet_pos(pos)) for word, pos in pos_tags[:20]]

pos_counts.plot(20, title="POS Frequency Distribution")
plt.show()

sentences = list(jieba.cut(moby_dick_text))

sentiment_scores = [SnowNLP(sentence).sentiments for sentence in sentences]

average_sentiment = sum(sentiment_scores) / len(sentiment_scores)

threshold = 0.5

overall_sentiment = "positive" if average_sentiment > threshold else "negative"

print(f"average score: {average_sentiment}")
print(f"sentiment: {overall_sentiment}")
