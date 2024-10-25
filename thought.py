"""
Please answer the following thought questions. These are questions that help you reflect
on the process of building this chatbot and about ethics. 

We are expecting at least three complete sentences for each question for full credit. 

Each question is worth 2.5 points. 
"""

######################################################################################
"""
QUESTION 1 - Anthropomorphizing

Is there potential for users of your chatbot possibly anthropomorphize 
(attribute human characteristics to an object) it? 
What are some possible ramifications of anthropomorphizing chatbot systems? 
Can you think of any ways that chatbot designers could ensure that users can easily 
distinguish the chatbot responses from those of a human?
"""

Q1_your_answer = """

Yes, especially if users are not aware of its internal workings, making it seem as if it is "thinking." Some ramifications of anthropomorphizing chatbot systems, particularly the most advanced ones such as ChatGPT, include inordinate emotional attachment to its responses, which can prevent people who are, for example, seeking help in terms of mental health from accessing actual resources. Chatbot designers can continually signpost in the chatbot responses its role as a programmed machine with automated responses; use language that reinforces on some level the artificiality of the interaction between human and chatbot.

"""

######################################################################################

"""
QUESTION 2 - Privacy Leaks

One of the potential harms for chatbots is collecting and then subsequently leaking 
(advertly or inadvertently) private information. Does your chatbot have risk of doing so? 
Can you think of ways that designers of the chatbot can help to mitigate this risk? 
"""

Q2_your_answer = """

Yes, this chatbot has a risk of doing this. It collects information about people's movie preferences, which, especially if the movies are quite risqué or perhaps could be construed as embarrassing, they might not want to be rendered public. Designers of chatbots can ensure that data of this sort is not tied to particular users' identities, and perhaps avoid storing it after the sessions are concluded.

"""

######################################################################################

"""
QUESTION 3 - Classifier 

When designing your chatbot, you had the choice of using a lexicon-based sentiment analysis module 
or a statistical module (logistic regression). Which one did you end up choosing for your 
system. Why? 
"""

Q3_your_answer = """

We ended up using the logistic regression module. The problem we identified with the rule-based module was that simply counting words did not at all seem to adequately address the nuance that we expected our users to express in their opinions. Logistic regression, by contrast, has the potential to learn relationships that might escape the rule-based module. 

"""

"""
QUESTION 4 - Reflection 

You just built a frame-based dialogue system using a combination of rule-based and machine learning 
approaches. Congratulations! Reflect on the advantages and disadvantages of this paradigm. 
"""

Q4_your_answer = """

The frame-based dialogue system has some advantages -- it allows a human to specify with great detail the various cases and conditions under which certain responses are made, honing the interaction with precision. However, the disadvantages include that that meticulous plotting takes much effort and also is very domain-specific -- the illusion of linguistic dynamism and interaction is shattered when the user discovers a way to expose the artifice of the chatbot. Looking beyond frame-based approaches, the end-to-end approach seems, as we all have seen in ChatGPT, to provide that dynamism to a greater extent.

"""