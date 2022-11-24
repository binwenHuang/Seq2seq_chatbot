# Zakaria Daud Graduate Seq2seq chatbot project

# cornell_Data_Utils.py preprocesses the text data from Cornell Movie Dialog Corpus
# The dataset consists of two files movie_conversations.txt and movie_lines.txt
# movie_lines.txt consists of lines from movie scripts identified by line num
# e.g L366 +++$+++ u0 +++$+++ m0 +++$+++ BIANCA +++$+++ You're sweet.
# movie_conversations.txt is a file describing which lines are part of conversation
# e.g u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L200', 'L201', 'L202', 'L203']


#Imported libraries
import re

#Function to replace abbreviated words with their full form
def clean_text(text):
    text = text.lower()
    
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"  ","",text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    
    return text

#Function to pair corpus lines into questions and responses
def paired_data(movie_line,movie_convo):
    #Open movie_lines.txt and movie_conversations.txt and split into indv lines
    m_lines = open(movie_line , encoding='utf-8',errors='ignore').read().split('\n')
    c_lines = open(movie_convo , encoding='utf-8',errors='ignore').read().split('\n')
    
    #Extract line numbers from movie_conversations.txt
    convo_line = []
    for lines in c_lines:
        _lines = lines.split(" +++$+++ ")[-1][1:-1].replace("'","").replace(" ","")
        convo_line.append(_lines.split(","))
    
    # Put lines from script in dictionary with line num as id
    id_line = {}
    for lines in m_lines:
        _lines = lines.split(" +++$+++ ")
        if len(_lines) == 5:
            id_line[_lines[0]] = _lines[4]

    questions = []
    answers = []
    
    #Put lines in the same conversation as questions and answers in list
    for line in convo_line:
        for i in range(len(line) -1):
            questions.append(id_line[line[i]])
            answers.append(id_line[line[i+1]])

    clean_questions = []
    clean_answers = []
    
    # Clean questions and responses by removing abbreviations
    for q in questions:
        clean_questions.append(clean_text(q))

    for a in answers:
        clean_answers.append(clean_text(a))

    return clean_questions,clean_answers

# Function to limit only questions and responses within specified length range
def data_shorting(max_length,min_length,clean_questions,clean_answers):
    short_questions_temp = []
    short_answers_temp = []
    shorted_q = []
    shorted_a = []
    
    i = 0
    for question in clean_questions:
        if len(question.split()) >= min_length and len(question.split()) <= max_length:
            short_questions_temp.append(question)
            short_answers_temp.append(clean_answers[i])
        i += 1

    i=0
    for answer in short_answers_temp:
        if len(answer.split()) >= min_length and len(answer.split()) <= max_length:
            shorted_a.append(answer)
            shorted_q.append(short_questions_temp[i])
        i +=1

    return shorted_q,shorted_a

# Function to count number of occurences of each word in questions & responses. Bag of words
# Create index for words and only accept words that occur more than min threshold of times
def data_vocabs(shorted_q,shorted_a,threshold):
    vocab = {}
    
    # Count all the words that occur in questions and answers to create full vocab
    for question in shorted_q:
        for words in question.split():
            if words not in vocab:
                vocab[words] = 1
            else:
                vocab[words] +=1

    for answer in shorted_a:
        for words in answer.split():
            if words not in vocab:
                vocab[words] = 1
            else:
                vocab[words] +=1
    
    # Count words that occur in questions
    questions_vocabs = {}
    for answer in shorted_q:
        for words in answer.split():
            if words not in questions_vocabs:
                questions_vocabs[words] = 1
            else:
                questions_vocabs[words] +=1
    
    # Count words that occur in answers
    answers_vocabs = {}
    for answer in shorted_a:
        for words in answer.split():
            if words not in answers_vocabs:
                answers_vocabs[words] = 1
            else:
                answers_vocabs[words] +=1
    
    # Represent each word that occurs in vocabulary with number if occurs more than minimum frequency
    vocabs_to_index = {}
    word_num = 0
    for word, count in vocab.items():
        if count >= threshold:
            vocabs_to_index[word] = word_num
            word_num += 1
    
    # Special tokens that are part of vocabulary
    codes = ['<PAD>','<EOS>','<UNK>','<GO>']
    
    # Represent special tokens as numbers and add them to questions & ans vocab
    for code in codes:
        vocabs_to_index[code] = len(vocabs_to_index)+1
    
    for code in codes:
        questions_vocabs[code] = len(questions_vocabs)+1

    for code in codes:
        answers_vocabs[code] = len(answers_vocabs)+1
    
    # Reverse dictionary for number to words
    index_to_vocabs = {v_i: v for v, v_i in vocabs_to_index.items()}

    return vocab,vocabs_to_index,index_to_vocabs,len(questions_vocabs),len(answers_vocabs)


# Function to create number representations for questions and responses using vocab_to_index dictionary
def data_int(shorted_q,shorted_a,vocabs_to_index):
    
    questions_int = []
    for question in shorted_q:
        ints = []
        for word in question.split():
            if word not in vocabs_to_index:
                ints.append(vocabs_to_index['<UNK>'])
            else:
                ints.append(vocabs_to_index[word])
        questions_int.append(ints)

    answers_int = []
    for answer in shorted_a:
        ints = []
        for word in answer.split():
            if word not in vocabs_to_index:
                ints.append(vocabs_to_index['<UNK>'])
            else:
                ints.append(vocabs_to_index[word])
        answers_int.append(ints)

    return questions_int,answers_int

# Wrapper function to preprocess all the data by calling previous functions
def preparing_data(movie_line,movie_convo,max_length,min_length,threshold):
    
    # Pair questions and responses
    clean_questions,clean_answers = paired_data(movie_line,movie_convo)
    
    # Remove questions and answers that are outside specified length range
    shorted_q,shorted_a = data_shorting(max_length,min_length,clean_questions,clean_answers)
    
    # Count all words that occur and return dict with number representation for each word
    vocab,vocabs_to_index,index_to_vocabs,question_vocab_size,answer_vocab_size = data_vocabs(shorted_q,shorted_a,threshold)
    
    # Add EOS token to end of all answers
    for i in range(len(shorted_a)):
        shorted_a[i] += ' <EOS>'
    
    # Convert questions and responses to numerical form
    questions_int,answers_int = data_int(shorted_q,shorted_a,vocabs_to_index)

    return questions_int,answers_int,vocabs_to_index,index_to_vocabs,question_vocab_size,answer_vocab_size






