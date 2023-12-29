# Sentimentally   

Sentimentally is an updated version of my college project, a neural network trained on the IMDb dataset, with the intention of training a decent sentiment analysis model for other kinds of text.
The aim is to build a tool for constructing neural networks, for potentially other projects in the future, therefore we will ban ourselves from using libraries such as tensorflow. We will however be kind to ourselves and let ourselves use data processing libraries such as pandas and numpy. We will also save ourselves the grief of text preprocessing, and pass a lot of the work to the Regex library.   

## The idea

We will aim to build a fairly simple model, using TF-IDF as our word embedding strategy. This will then be passed on to our own model, to be parsed into a single float value between zero and one, that will theoretically be a guess as to whether the review is negative or positive. 

## 1. Understanding the IMDb Dataset:
The IMDb dataset consists of movie reviews labeled as positive or negative sentiments. It contains 25k positive and 25k negative reviews. It can be accessed [here](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?resource=download)   


## 2. Data preprocessing
The first real step to take is to clean our data of punctation, symbols, anything that's not forming a word. Another step is to ensure all letters are lowercase. This way we can create a dictionary, that will let us create our TF-IDF vector representations. The dictionary will map each word to a numerical value.
Dictionary fetched from [here](https://github.com/dwyl/english-words?tab=readme-ov-file)

## 3. Tokenising
We can use our dictionary to tokenise each review in our dataset, replacing each word by its numerical representation. From here we can create a bag of words vector for each review.

## 4. TF-IDF
The next step is using our bag of words representation of each review into a TF-IDF representation. TF-IDF stansds for Term Frequency Inverse Document Frequency, and it measures how important a term is within a document relative to a corpus of documents. More information can be found on its [Wikipedia page](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)

## 5. Building the model
The main goal will be to have the model fully customisable, but for now we will follow the defaults from my college project. This means using sigmoid as our activation, and a binary cross-entropy loss function. 
