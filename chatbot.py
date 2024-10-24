"""
Class that implements the chatbot for CSCI 375's Midterm Project. 

Please follow the TODOs below with your code. 
"""
import numpy as np
import argparse
import joblib
import re  
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
import nltk 
from collections import defaultdict, Counter
from typing import List, Dict, Union, Tuple

import util

class Chatbot:
    """Class that implements the chatbot for CSCI 375's Midterm Project"""

    def __init__(self):
        # The chatbot's default name is `moviebot`.
        self.name = 'auteur' # TODO: Give your chatbot a new name.

        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, self.ratings = util.load_ratings('data/ratings.txt')
        
        # Load sentiment words 
        self.sentiment = util.load_sentiment_dictionary('data/sentiment.txt')

        # Train the classifier
        self.train_logreg_sentiment_classifier()

        # TODO: put any other class variables you need here 

    ############################################################################
    # 1. WARM UP REPL                                                          #
    ############################################################################

    def intro(self) -> str:
        """
        Return a string to use as your chatbot's description for the user.

        Consider adding to this description any information about what your
        chatbot can do and how the user can interact with it.
        """

        # TODO: delete and replace the line below
        return """
        this is a chatbot which will provide you with film recommendations based on your proclivities.
        give me a film and tell me how you felt about it. and maybe four more after that. 

        to exit: write ":quit" (or press Ctrl-C to force the exit)
        """

    def greeting(self) -> str:
        """Return a message that the chatbot uses to greet the user."""

        greeting_message = "what's up?"

        return greeting_message

    def goodbye(self) -> str:
        """
        Return a message that the chatbot uses to bid farewell to the user.
        """

        goodbye_message = "until next time..."

        return goodbye_message

    def debug(self, line):
        """
        Returns debug information as a string for the line string from the REPL

        No need to modify this function. 
        """
        return str(line)

    ############################################################################
    # 2. Extracting and transforming                                           #
    ############################################################################

    def process(self, line: str) -> str:
        """
        Process a line of input from the REPL and generate a response.

        This is the method that is called by the REPL loop directly with user
        input.

        You should delegate most of the work of processing the user's input to
        the helper functions you write later in this script.

        Takes the input string from the REPL and call delegated functions that
          1) extract the relevant information, and
          2) transform the information into a response to the user.

        Example:
          resp = chatbot.process('I loved "The Notebook" so much!!')
          print(resp) // prints 'So you loved "The Notebook", huh?'
        
        Arguments: 
            - line (str): a user-supplied line of text
        
        Returns: a string containing the chatbot's response to the user input

        Hints: 
            - We recommend doing this function last (after you've completed the
            helper functions below)
            - Try sketching a control-flow diagram (on paper) before you begin 
            this 
            - We highly recommend making use of the class structure and class 
            variables when dealing with the REPL loop. 
            - Feel free to make as many helper funtions as you would like 
        """

        response = "I (the chatbot) processed '{}'".format(line)

        return response

    def extract_titles(self, user_input: str) -> List[str]:
        """
        Extract potential movie titles from the user input.

        - If there are no movie titles in the text, return an empty list.
        - If there is exactly one movie title in the text, return a list
        containing just that one movie title.
        - If there are multiple movie titles in the text, return a list
        of all movie titles you've extracted from the text.

        Example 1:
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'I do not like any movies'))
          print(potential_titles) // prints []

        Example 2:
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'I liked "The Notebook" a lot.'))
          print(potential_titles) // prints ["The Notebook"]

        Example 3: 
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'There are "Two" different "Movies" here'))
          print(potential_titles) // prints ["Two", "Movies"]                              
    
        Arguments:     
            - user_input (str) : a user-supplied line of text

        Returns: 
            - (list) movie titles that are potentially in the text

        Hints: 
            - What regular expressions would be helpful here? 
        """
        
        # regex pattern to get anything within double quotes
        regex = r'"([^"]+)"'
        
        # match user input on the regex
        tuples = re.findall(regex, user_input)
        
        # get a list of the matches
        films = [fst for fst in tuples]
        
        return films
    

    def find_movies_idx_by_title(self, title:str) -> List[int]:
        """ 
        Given a movie title, return a list of indices of matching movies
        The indices correspond to those in data/movies.txt.

        - If no movies are found that match the given title, return an empty
        list.
        - If multiple movies are found that match the given title, return a list
        containing all of the indices of these matching movies.
        - If exactly one movie is found that matches the given title, return a
        list that contains the index of that matching movie.

        Example 1:
          ids = chatbot.find_movies_idx_by_title('Titanic')
          print(ids) // prints [1359, 2716]

        Example 2:
          ids = chatbot.find_movies_idx_by_title('Twelve Monkeys')
          print(ids) // prints [31]

        Example 3: 
          ids = chatbot.find_movies_idx_by_title("Michael Collins (1996)")
          print(ids) // prints [800]

        Arguments:
            - title (str): the movie title 

        Returns: 
            - a list of indices of matching movies

        Hints: 
            - You should use self.titles somewhere in this function.
              It might be helpful to explore self.titles in scratch.ipynb
            - You might find one or more of the following helpful: 
              re.search, re.findall, re.match, re.escape, re.compile
            - Our solution only takes about 7 lines. If you're using much more 
            than that try to think of a more concise approach 
            """
        
        # initialize empty array of movies
        ret = []
        
        # look for matching movies in self.titles and add them to ret
        for i in range(len(self.titles)):
            if title in self.titles[i][0]:
                ret.append(i)
        
        # return the resulting list of matched movies
        return ret


    def disambiguate_candidates(self, clarification: str, candidates: list) -> List[int]: 
        """
        Given a list of candidate movies that the user could be
        talking about (represented as indices), and a string given by the user
        as clarification (e.g. in response to your bot saying "Which movie did
        you mean: Titanic (1953) or Titanic (1997)?"), use the clarification to
        narrow down the list and return a smaller list of candidates (hopefully
        just 1!)

        - If the clarification uniquely identifies one of the movies, this
        should return a 1-element list with the index of that movie.
        - If the clarification does not uniquely identify one of the movies, this 
        should return multiple elements in the list which the clarification could 
        be referring to. 

        Example 1 :
          chatbot.disambiguate_candidates("1997", [1359, 2716]) // should return [1359]

          Used in the middle of this sample dialogue 
              moviebot> 'Tell me one movie you liked.'
              user> '"Titanic"'
              moviebot> 'Which movie did you mean:  "Titanic (1997)" or "Titanic (1953)"?'
              user> "1997"
              movieboth> 'Ok. You meant "Titanic (1997)"'

        Example 2 :
          chatbot.disambiguate_candidates("1994", [274, 275, 276]) // should return [274, 276]

          Used in the middle of this sample dialogue
              moviebot> 'Tell me one movie you liked.'
              user> '"Three Colors"''
              moviebot> 'Which movie did you mean:  "Three Colors: Red (Trois couleurs: Rouge) (1994)"
                 or "Three Colors: Blue (Trois couleurs: Bleu) (1993)" 
                 or "Three Colors: White (Trzy kolory: Bialy) (1994)"?'
              user> "1994"
              movieboth> 'I'm sorry, I still don't understand.
                            Did you mean "Three Colors: Red (Trois couleurs: Rouge) (1994)" or
                            "Three Colors: White (Trzy kolory: Bialy) (1994)" '
    
        Arguments: 
            - clarification (str): user input intended to disambiguate between the given movies
            - candidates (list) : a list of movie indices

        Returns: 
            - a list of indices corresponding to the movies identified by the clarification

        Hints: 
            - You should use self.titles somewhere in this function 
            - You might find one or more of the following helpful: 
              re.search, re.findall, re.match, re.escape, re.compile
        """
        clarified_list = []

        for movie_index in candidates:
            line = self.titles[movie_index][0]
            matches = re.findall(clarification, line)
            for match in matches:
                if match != None:
                    clarified_list.append(movie_index)
        return clarified_list
    

    ############################################################################
    # 3. Sentiment                                                             #
    ########################################################################### 

    def predict_sentiment_rule_based(self, user_input: str) -> int:
        """
        Predict the sentiment class given a user_input

        In this function you will use a simple rule-based approach to 
        predict sentiment. 

        Use the sentiment words from data/sentiment.txt which we have already 
        loaded for you in self.sentiment. 
        
        Then count the number of tokens that are in the positive sentiment category 
        (pos_tok_count) and negative sentiment category (neg_tok_count).

        This function should return 
        -1 (negative sentiment): if neg_tok_count > pos_tok_count
        0 (neutral): if neg_tok_count is equal to pos_tok_count
        +1 (postive sentiment): if neg_tok_count < pos_tok_count

        Example:
          sentiment = chatbot.predict_sentiment_rule_based('I LOVE "The Titanic"'))
          print(sentiment) // prints 1
        
        Arguments: 
            - user_input (str) : a user-supplied line of text
        Returns: 
            - (int) a numerical value (-1, 0 or 1) for the sentiment of the text

        Hints: 
            - Take a look at self.sentiment (e.g., in scratch.ipynb)
            - Remember we want the count of *tokens* not *types*
        """
        pos_tok_count = 0
        neg_tok_count = 0
        
        user_input_tokens = re.sub(r'[^\w\s]', '', user_input)
        
        for tok in user_input_tokens.lower().split():
            sentiment = self.sentiment.get(tok)
            if tok in self.sentiment:
                if self.sentiment[tok] == "pos":
                    print("pos word: " + tok)
                    pos_tok_count += 1
                else:
                    neg_tok_count += 1
                    print("neg word: " + tok)
                    
        if pos_tok_count > neg_tok_count:
            return 1
        elif pos_tok_count == neg_tok_count:
            return 0
        else:
            return -1
 

    def train_logreg_sentiment_classifier(self):
        """
        Trains a bag-of-words Logistic Regression classifier on the Rotten Tomatoes dataset 

        You'll have to transform the class labels (y) such that: 
            -1 inputed into sklearn corresponds to "rotten" in the dataset 
            +1 inputed into sklearn correspond to "fresh" in the dataset 
        
        To run call on the command line: 
            python3 chatbot.py --train_logreg_sentiment

        Hints: 
            - You do not need to write logistic regression from scratch (you did that in HW3). 
            Instead, look into the sklearn LogisticRegression class. We recommend using scratch.ipynb
            to get used to the syntax of sklearn.LogisticRegression on a small toy example. 
            - Review how we used CountVectorizer from sklearn in this code
                https://github.com/cs375williams/hw3-logistic-regression/blob/main/util.py#L193
            - You'll want to lowercase the texts
            - Our solution uses less than about 10 lines of code. Your solution might be a bit too complicated.
            - We achieve greater than accuracy 0.7 on the training dataset. 
        """ 
        #load training data  
        texts, y_str = util.load_rotten_tomatoes_dataset()
        
        # transform class labels to ints
        y = [-1 if elem=="Rotten" else 1 for elem in y_str]
        print(y)
        
        # lowercase all the texts
        texts = [text.lower() for text in texts]

        
        # fit a count vectorizer to learn the vocab
        self.count_vectorizer = CountVectorizer(min_df=20,stop_words='english',max_features=1000)  
        self.X = self.count_vectorizer.fit_transform(texts).toarray()
        
        # train a logistic regression classifier on X and y
        self.model = LogisticRegression()
        self.model.fit(self.X, y)
        pass 


    def predict_sentiment_statistical(self, user_input: str) -> int: 
        """ 
        Uses a trained bag-of-words Logistic Regression classifier to classify the sentiment

        In this function you'll also use sklearn's CountVectorizer that has been 
        fit on the training data to get bag-of-words representation.

        Example 1:
            sentiment = chatbot.predict_sentiment_statistical('This is great!')
            print(sentiment) // prints 1 

        Example 2:
            sentiment = chatbot.predict_sentiment_statistical('This movie is the worst')
            print(sentiment) // prints -1

        Example 3:
            sentiment = chatbot.predict_sentiment_statistical('blah')
            print(sentiment) // prints 0

        Arguments: 
            - user_input (str) : a user-supplied line of text
        
        Returns: int 
            -1 if the trained classifier predicts -1 
            1 if the trained classifier predicts 1 
            0 if the input has no words in the vocabulary of CountVectorizer (a row of 0's)

        Hints: 
            - Be sure to lower-case the user input 
            - Don't forget about a case for the 0 class! 
        """                                           
        
        # preset sentiment to 0
        sentiment = 0
        
        # use fitted vectorizer to eval the user input into bag of words
        vectorized = self.count_vectorizer.transform([user_input]).toarray()
        
        # if at least 1 word in the input is in the vocab, then predict
        if np.any(vectorized):
            # predict!
            sentiment = self.model.predict(vectorized)[0]

        return sentiment


    ############################################################################
    # 4. Movie Recommendation                                                  #
    ############################################################################

    def recommend_movies(self, user_ratings: dict, num_return: int = 3) -> List[str]:
        """
        This function takes user_ratings and returns a list of strings of the 
        recommended movie titles. 

        Be sure to call util.recommend() which has implemented collaborative 
        filtering for you. Collaborative filtering takes ratings from other users
        and makes a recommendation based on the small number of movies the current user has rated.  

        This function must have at least 5 ratings to make a recommendation. 

        Arguments: 
            - user_ratings (dict): 
                - keys are indices of movies 
                  (corresponding to rows in both data/movies.txt and data/ratings.txt) 
                - values are 1, 0, and -1 corresponding to positive, neutral, and 
                  negative sentiment respectively
            - num_return (optional, int): The number of movies to recommend

        Example: 
            bot_recommends = chatbot.recommend_movies({100: 1, 202: -1, 303: 1, 404:1, 505: 1})
            print(bot_recommends) // prints ['Trick or Treat (1986)', 'Dunston Checks In (1996)', 
            'Problem Child (1990)']

        Hints: 
            - You should be using self.ratings somewhere in this function 
            - It may be helpful to play around with util.recommend() in scratch.ipynb
            to make sure you know what this function is doing. 
        """ 
        
        # make sure precondition is satisfied
        assert len(user_ratings.keys()) >= 5
        
        # put user ratings at correct indices in an array of all user ratings
        user_array = np.zeros(len(self.titles))
        for key in user_ratings.keys():
            user_array[key] = user_ratings[key]
        
        # call util.recommend to get indices for recommendations
        indices = util.recommend(user_array, self.ratings, num_return)
        
        # get titles at indices
        films = [self.titles[i][0] for i in indices]
        
        # return those titles as the recommendations
        return films


    ############################################################################
    # 5. Open-ended                                                            #
    ############################################################################

    def function1(self, user_input: str):
        """
        This function takes in a user input string.
        It identifies movies without quotation marks, ignoring incorrect capitalization, and rejects titles which are strict substrings of other titles.
        It returns a list of the movies identified in the user input string.
        """
        # initialize empty array of movies
        movies = []
        indices = []
       
        # regex to capture title
        regex = r'(.+)(?:\(\d+\))'
        
        # look for matching movies in self.titles and add them to ret
        for i in range(len(self.titles)):
                
            # match on title, discard year
            matches = re.findall(regex, self.titles[i][0].lower())
            
            # continue if no matches
            if len(matches) == 0:
                continue
            
            # strip the whitespace
            title = matches[0].strip()
            
            regex2 = r'\b' + re.escape(title) + r'\b'
                
            # detect title in user string
            t = re.findall(regex2, user_input.lower())
            if len(t) > 0:
                # add actual title to list if it's not a substring of already-added film
                if all([t[0] not in film for film in movies]):
                    movies.append(self.titles[i][0])
                    indices.append(i)
        print(movies)
        # return the resulting list of matched movies
        return indices

    def function2(self, title: str):
        """
        This function takes in a title. 
        It matches on self.titles, accounting for articles (e.g. matching An American in Paris with American in Paris, An).
        It returns a list of the matched indices.
        """
        # capture the title split into not-article and article
        regex = r'(.+), ([An|The|A]+)[ |^\)]?'
        
        # match on all titles
        remove_parens = [re.findall(r'^[^\(]*', item[0]) for item in self.titles]
        regexed = [re.findall(regex, item[0]) for item in remove_parens]
        
        # transform into article + not-article form
        titles = [match[0][1] + " " + match[0][0] if len(match)>0 else "" for match in regexed]
        
        # initialize empty array of movies
        indices = []
        
        # look for matching movies in self.titles and add them to ret
        for i in range(len(titles)):
            if title in titles[i]:
                indices.append(i)
        
        # return the resulting list of matched movies
        return indices
            

    def function3(): 
        """
        Any additional functions beyond two count towards extra credit  
        """
        pass 


if __name__ == '__main__':
    print('To run your chatbot in an interactive loop from the command line, '
          'run:')
    print('    python3 repl.py')



