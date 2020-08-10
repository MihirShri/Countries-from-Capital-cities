"""
Author @ Mihir_Srivastava
Dated - 18-07-2020
File - Finding_countries_from_capitals
Aim - To find a country given its capital city, relation between a country and its capital, using word embeddings and
cosine similarity.
"""

# Import necessary libraries
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# Read csv file
df = pd.read_csv('capitals.txt', sep=' ', names=['city1', 'country1', 'city2', 'country2'], skiprows=1)

# Load word embeddings
word_embeddings = pickle.load(open("word_embeddings_subset.p", "rb"))


# A function to get the most similar country given a capital city, relation between another country and its capital and
# the word embeddings.
def get_similar(city1, country1, city2, embeddings):
    # Create a set of the given cities and country
    group = {city1, country1, city2}

    # Get the word embeddings of the given cities and country
    city1_em = embeddings[city1]
    country1_em = embeddings[country1]
    city2_em = embeddings[city2]

    # Using the similarity between city1 and country, generate a word vector similar to that of city2
    vec = country1_em - city1_em + city2_em

    similarity = -1
    country2 = ''

    # For each word in the word embeddings dictionary
    for word in embeddings.keys():
        if word not in group:
            word_em = embeddings[word]

            # Find the cosine similarity between the word vector of each word and the word vector obtained earlier
            cosine_sim = cosine_similarity([vec], [word_em])

            # Get the word with maximum similarity
            if cosine_sim > similarity:
                similarity = cosine_sim
                country2 = (word, similarity)

    return country2


# A function to get the accuracy of our model
def get_accuracy(word_embeddings, data):
    correct = 0

    # For each row in our DataFrame
    for i, row in data.iterrows():
        # Get the capital cities and countries
        city1 = row['city1']
        country1 = row['country1']
        city2 = row['city2']
        country2 = row['country2']

        # Predict country 2 (country whose capital is city2)
        pred = get_similar(city1, country1, city2, word_embeddings)[0]

        # Get the total number of correct predictions
        if pred == country2:
            correct += 1

    m = len(data)

    return float(correct)/m


print()
print('accuracy: ', get_accuracy(word_embeddings, df))
print()
city1 = input('Enter city1: ')
country1 = input('Enter country1: ')
city2 = input('Enter the capital city whose country you want to know: ')
print()
print(city2 + ' is the capital of ' + get_similar(city1, country1, city2, word_embeddings)[0])
