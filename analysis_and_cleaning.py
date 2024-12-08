import pandas as pd
import matplotlib.pyplot as plt
import re
import spacy
import emoji 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from spacy.cli import download


class anaylizer_cleaner:
    
    #Download spacy model for POS Tagging
    #considering the size of the dataset, using en_core_web_md instead of en_core_web_sm gives better results.
    nlp = spacy.load("en_core_web_md")

    def analyze_and_clean():
        download('en_core_web_md')

        #Read csv file
        review_data = pd.read_csv("datasets/spotify_user_review.csv")


        #### DATASET ANALYSIS ####

        print(review_data.head(10))        #first 10 value of review_data

        print(review_data.info())          #information table of review_data
        #According to this analysis, there are 2 columns in the dataset. Therese columns dtypes are object

        print(review_data.describe())      #statistics of review_data
        #According to this analysis, there are 14000 total data in the Review column. 13921 of the comments are unique, meaning some comments are repeated. The word "Too many ads" is the most repeated comment, repeated 17 times.
        #Label column: There are 14000 total data. There are 2 types of labels. The "Negative" label was used 7558 times, more than the positives.

        #Since changes will be made to the dataset, proceeding with a copy ensures safer analysis.
        copy_data = review_data.copy()

        print(copy_data.isnull().sum())  #checking for missing data
        #There are no null values in both column.


        print(copy_data.duplicated(subset=["Review"]).sum())  #control of repeated data in Review column
        #There are 79 duplicate review. These comments are approximately 0.5% of the dataset. Therefore, deleting repeated comments does not have any negative effects. 
        #Also, learning the same information does not change the decision of the model, but it can cause overfitting.

        copy_data = copy_data.drop_duplicates(subset=["Review"], keep="first")  #deleting duplicate data. With keep="first" the first visible record is kept and the others are deleted.
        print(copy_data.duplicated(subset=["Review"]).sum())  #control of repeated data again
        #Drop operation successful, no duplicate values left



        #Determining the categories of label types
        label_types = copy_data["label"].unique()
        print(f"Categories of label types: {label_types}")  #There are 2 label types. Positive and negative


        #Number of each label type
        label_type_numbers = copy_data["label"].value_counts()
        print(f"Number of each label \n{label_type_numbers}")


        ##Different color charts are created based on the label type for better visualization.
        colors = ['red' if label == 'NEGATIVE' else 'green' for label in label_type_numbers.index]

        label_type_numbers.plot(kind="bar", title="Number of each Label", figsize=(10, 6), color=colors)
        plt.ylabel("count")
        plt.xlabel("Label type")
        plt.show()


        copy_data = anaylizer_cleaner.remove_emojis_from_reviews(copy_data, 'Review')

        ##Call the function and get the results
        copy_data = anaylizer_cleaner.apply_preprocessings(copy_data)
        copy_data.to_csv("copy_data.csv", index=False) 

        #The cleaning process is checked.
        print(copy_data[["Review", "cleaned_review"]].head())

        print(copy_data.head(15))




    #### TEXT PREPROCESSING ####


    #There are emojis in the comments in the dataset. These emojis negatively affect the results of operations such as text cleaning and TF-IDF. 
    #Therefore, emojis were removed first, and then other text cleaning operations were performed.

    #Function to count and remove emojis
    def remove_emojis_from_reviews(data, column):
    
        initial_emoji_count = 0         #count the number of initial emojis

        for review in data[column]:
            initial_emoji_count += emoji.emoji_count(review)

        print(f"Initial emoji count in dataset: {initial_emoji_count}")

        
        data[column] = data[column].apply(lambda x: emoji.demojize(x))  #remove emojis using demojize


        #Loop for controlling remove emojis
        final_emoji_count = 0           #count the number of emojis after removal

        for review in data[column]:
            final_emoji_count += emoji.emoji_count(review)
        
        print(f"Final emoji count: {final_emoji_count}")

        return data


    # Text cleaning function
    def text_cleaning(text):
        
        text = re.sub(r'\d+', '', text)             # Remove numbers
        text = re.sub(r'[^\w\s]', '', text)         # Remove punctuation
        text = re.sub(r'http\S+', '', text)         # Remove URL
        text = re.sub(r'(.)\1{2,}', r'\1', text)    # Remove excessive repetitive characters
        text = re.sub(r'[^\x00-\x7F]+', '', text)   # Remove non-Unicode characters
        text = text.lower()                         # Convert to lowercase
        text = text.strip()                         # Clean unnecessary space
        
        return text



    #Tokenization and stopwords removal function
    def tokenize_and_remove_stopwords(text):
        #Split words into tokens.
        tokens = word_tokenize(text)      #Word tokenization was done for a more detailed analysis
        
        #Removes words with low meaning from tokens.
        set_stopwords = set(stopwords.words('english'))     #set English stopwords
        new_tokens = [word for word in tokens if word not in set_stopwords]
        
        return new_tokens



    #POS tagging function converts tags into a format that WordNetLemmatizer can understand.
    #WordNetLemmatizer accepts words as nouns. It can't separate other types of words into the correct root. Therefore, POS tagging was done before the lemmatization step.
    def pos_tagging(tag):

        if tag.startswith("J"):    #for adjective
            return wordnet.ADJ
        
        elif tag.startswith("R"):  #for adverb
            return wordnet.ADV

        elif tag.startswith("V"):  #for verb
            return wordnet.VERB
        
        else:                      #for noun. Noun is default value
            return wordnet.NOUN

    #Lemmatization function
    def lemmatize_text(new_tokens):

        lemmatizer = WordNetLemmatizer()    #initialize the WordNetLemmatizer. Because it helps normalize text for analysis by reducing words to their lemma
        lemmatized_tokens = []              #lemmatize every tokens in the list of new_tokens

        #Spacy was used for more accurate analysis. It makes POS tagging appropriate to the context of the word
        spacy_tags = [(word.text, word.pos_) for word in anaylizer_cleaner.nlp(" ".join(new_tokens))]
        
        for token, pos_tag in spacy_tags:
            
            wordnet_tag = anaylizer_cleaner.pos_tagging(pos_tag)   #convert Spacy POS to WordNet POS tags
            lemmatized_tokens.append(lemmatizer.lemmatize(token, wordnet_tag))  #lemmatization process based on POS Label

        return lemmatized_tokens



    #Merge preprocessing steps function
    def combined_preprocessing(text): 

        text = anaylizer_cleaner.text_cleaning(text)
        new_tokens = anaylizer_cleaner.tokenize_and_remove_stopwords(text)
        lemmatized_tokens = anaylizer_cleaner.lemmatize_text(new_tokens)

        return lemmatized_tokens



    #Function that applies the cleaned text to the entire dataset
    def apply_preprocessings(data):
        
        data["cleaned_review"] = data["Review"].apply(anaylizer_cleaner.combined_preprocessing) 
        
        return data 
    