# Tokenization

It is the process of breaking down sentences and set of words into smaller chunks or individual words. These individual parts are knows as tokens and the process of generating tokens is called tokenization.

This process helps in interpreting the sentences easier as it would help the computer to understand the individual words or phrases easily and thus interpret the given data and produce an output. It would help the computer understand the meaning by analysing the sequence of words.

eg: I am coming to Chennai could be tokenized into ‘I’, ’am’, ’coming’, ’to’, ’Chennai’.

However there’s a lot more nuances that need to considered while tokenizing statements to get the best results.

- We have to consider upper and lower case letters when tokenizing and make sure they are considered as the same word.
- We have to check for punctuation marks and make sure to sort those properly.
- There’s words like I’m or don’t where they actually represent 2 words. These have to be accounted and considered either as 2 different words or I am and I’m should be considered as the same term and not as ‘I’ ‘am’ for ‘I am’ and I’m for I’m
- Same kinds of words representing singular plural or superlative forms shouldn’t be considered as different words.
- Considering commonly used phrases as a single token rather than as individual words also helps in increasing the accuracy.

# Word Embeddings

It is the process of converting the tokens into a set of numbers(vectors) which the computer can understand and identify to interpret into a set of useful data.

Say the computer has to differentiate between words like India, Jonathan, Ram and Australia. 2 of these are countries and 2 are names of people but just by assigning each of them 1,2,3 and 4 wouldn’t help the computer differentiate them as names and countries. So instead just like how we consider multiple parameters to sat describe a person we take multiple parameters to describe a word and give corresponding values to these parameters for each of these words.

Then the computer compares these parameters to predict whether the given word is a country or a name.

For country say the parameters are “Presence on the map” and “Presence of a government” and for man say the parameters are “health”, “Presence of silicon” and “ability to eat ice cream”

When we put values for all these parameters for these, we get:

- India :1,1,0.5,0.25,0
- Jonathan:0,0,1,0,1
- Ram:0.01,0,1,0,1
- Australia:1,1,0,0,0

So here using these parameters the computer learns which set of values corresponds to a country and man and thus helps in differentiating and understand their meanings.

So if it is given the word India, it would compare parameters like presence on map and presence of government as a way to identify similarities with Australia and thus declare both to be something similar.

## Word2Vec

There’s 2 kinds of Word2Vec: Common Bag of Words(CBOW) and Skip Gram.

This uses a neural network to predict the words and thus assign them the corresponding vector parameters and weights.

We take a sentence or a paragraph and divide it into set of say 3 words. We divide these 3 words and train the neural network to predict the remaining words. We do this for all the words in the sentence that it would be word embedded with parameters to identify how it works and the context of how it works.

CBOW divides the 3 words into 2 context words and 1 target word. So these 2 words are given as inputs to a neural network. These pass through the hidden layers and assign final values to the target output. Weights assigned in the first round wouldnt be right. So the result is back-propagated to train the neural network and improve its accuracy by identifying the right weights. At the end of this we have a properly functioning neural network to identify the word embeddings.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e4afd78b-0087-4624-9793-c69c4b7f38d8/Untitled.png)

Skip Gram uses 2 target words and one context word. The first word is taken as the context word and the network is trained by the same process but made to determine 2 words rather than 1 in CBOW.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d356a8fb-1f24-4e60-8092-7588aeeda5b4/Untitled.png)

Q2-

[https://colab.research.google.com/drive/1m4Pkd3cCGHexy39ExJhScjzp61W7tcj1?usp=sharing](https://colab.research.google.com/drive/1m4Pkd3cCGHexy39ExJhScjzp61W7tcj1?usp=sharing)

CSV file:[https://support.staffbase.com/hc/en-us/article_attachments/360009197031/username.csv](https://support.staffbase.com/hc/en-us/article_attachments/360009197031/username.csv)
