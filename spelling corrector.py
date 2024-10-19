import pandas as pd
from collections import Counter
import numpy as np

# Load dataset
data = pd.read_csv('spelling.csv')

# Data preprocessing
corpus = data['correct_spellings'].str.lower().tolist()
word_counts = Counter(corpus)

# Build vocabulary
vocabulary = set(corpus)

# Function to calculate edit distance between two words
def edit_distance(word1, word2):
    m, n = len(word1), len(word2)
    dp = np.zeros((m+1, n+1))
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]

# Function to find nearest correct word
def find_nearest_word(word):
    min_distance = float('inf')
    nearest_word = None
    for vocab_word in vocabulary:
        distance = edit_distance(word, vocab_word)
        if distance < min_distance:
            min_distance = distance
            nearest_word = vocab_word
    return nearest_word

# Testing
misspelled_words = ["acess", "writting", "peopl", "compter"]
for word in misspelled_words:
    corrected_word = find_nearest_word(word)
    print(f"Misspelled Word: {word}, Corrected Word: {corrected_word}")