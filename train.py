# Zakaria Daud Graduate Seq2seq chatbot project train.py

#Imported libararies
import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt 
import math
from tqdm import tqdm
import pickle
import os
from cornell_Data_Utils import preparing_data
from model import seq2seq_model,pad_sentence,get_accuracy,sentence_to_seq
import config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#tf.logging.set_verbosity(tf.logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Setting hyperparameters to values defined in config.py
BATCH_SIZE = config.BATCH_SIZE #128
RNN_SIZE = config.RNN_SIZE     #128
EMBED_SIZE = config.EMBED_SIZE #128
LEARNING_RATE = config.LEARNING_RATE #0.001
KEEP_PROB = config.KEEP_PROB #0.75
EPOCHS = config.EPOCHS       #500
MODEL_DIR = config.MODEL_DIR
SAVE_PATH = config.SAVE_PATH

# Directory to cornell movie dialog corpus lines and conversations made of lines
movie_line = 'Datasets/cornell movie-dialogs corpus/movie_lines.txt'
movie_convo = 'Datasets/cornell movie-dialogs corpus/movie_conversations.txt'

# Specify length range of accepted questions and responses
max_conversation_length = 5
min_conversation_length = 2
# Specify threshold for min number of times a word occurs to be part of vocab
min_frequency_words = 3

# Preprocess the data. Convert questions & responses to numbers using vocab_to_index dict
questions_int,answers_int,vocabs_to_index,index_to_vocabs,question_vocab_size,answer_vocab_size = preparing_data(movie_line,
	movie_convo,max_conversation_length,
	min_conversation_length,min_frequency_words)

vocab_size = len(index_to_vocabs)

# Save vocabs_to_index & index_to_vocabs dictionaries to pickle file
pickle.dump(vocabs_to_index, open("vocab2index.p", "wb"))
pickle.dump(index_to_vocabs, open("index2vocab.p", "wb"))

# Divide numerical questions and answers into train, test and validation data
train_data = questions_int[BATCH_SIZE:] # 22992 questions
test_data = answers_int[BATCH_SIZE:]    # 22992 responses
val_train_data = questions_int[:BATCH_SIZE] # 128
val_test_data = answers_int[:BATCH_SIZE]    # 128

pad_int = vocabs_to_index['<PAD>']

# Pad variable length sequences to length of longest sequence (5) with PAD token
val_batch_x,val_batch_len = pad_sentence(val_train_data,pad_int)
val_batch_y,val_batch_len_y = pad_sentence(val_test_data,pad_int)
# Convert to numpy array
val_batch_x = np.array(val_batch_x)
val_batch_y = np.array(val_batch_y)

no_of_batches = math.floor(len(train_data)//BATCH_SIZE) # 179 batches
round_no = no_of_batches*BATCH_SIZE

# Create seq2seq model with encoder decoder architecture described in model.py
input_data,target_data,input_data_len,target_data_len,lr_rate,keep_probs,inference_logits,cost,train_op = seq2seq_model(question_vocab_size,
	EMBED_SIZE,RNN_SIZE,KEEP_PROB,answer_vocab_size,
	BATCH_SIZE,vocabs_to_index)

# Sample sentence used to test ouput during training
translate_sentence = 'how are you'
translate_sentence = sentence_to_seq(translate_sentence, vocabs_to_index)

acc_plt = []
loss_plt = []

# Start tensorflow session to execute graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Train for specified number of epochs
    for epoch in range(EPOCHS):
        total_accuracy = 0.0
        total_loss = 0.0
        # Progress bar to show during training
        for bs in tqdm(range(0,round_no  ,BATCH_SIZE)):
          index = min(bs+BATCH_SIZE, round_no )
          
          #Pad train and test data with PAD token to fixed length seq
          batch_x,len_x = pad_sentence(train_data[bs:index],pad_int)
          batch_y,len_y = pad_sentence(test_data[bs:index],pad_int)
          batch_x = np.array(batch_x)
          batch_y = np.array(batch_y)
          # Train the model using batches to obtain prediction
          pred,loss_f,opt = sess.run([inference_logits,cost,train_op], 
                                      feed_dict={input_data:batch_x,
                                                target_data:batch_y,
                                                input_data_len:len_x,
                                                target_data_len:len_y,
                                                lr_rate:LEARNING_RATE,
                                                keep_probs:KEEP_PROB})
          # Calculate training accuracy and total loss
          train_acc = get_accuracy(batch_y, pred)
          total_loss += loss_f 
          total_accuracy+=train_acc
        
        
        total_accuracy /= (round_no // BATCH_SIZE)
        total_loss /=  (round_no//BATCH_SIZE)
        acc_plt.append(total_accuracy)
        loss_plt.append(total_loss)
        
        # Get predicted output for translate sentence
        translate_logits = sess.run(inference_logits, {input_data: [translate_sentence]*BATCH_SIZE,
                                         input_data_len: [len(translate_sentence)]*BATCH_SIZE,
                                         target_data_len: [len(translate_sentence)]*BATCH_SIZE,              
                                         keep_probs: KEEP_PROB,
                                         })[0]
        
        # Output Epochs, average loss, avg accuracy and output to translate sentence
        print('Epoch %d,Average_loss %f, Average Accucracy %f'%(EPOCHS+1,total_loss,total_accuracy))
        print('  Inputs Words: {}'.format([index_to_vocabs[i] for i in translate_sentence]))
        print('  Replied Words: {}'.format(" ".join([index_to_vocabs[i] for i in translate_logits])))
        print('\n')
        saver = tf.train.Saver() 
        saver.save(sess,MODEL_DIR+"/"+SAVE_PATH)

# Plot accuracy vs epoch
plt.plot(range(EPOCHS),acc_plt)
plt.title("Change in Accuracy")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

# Plot loss vs epoch
plt.plot(range(EPOCHS),loss_plt)
plt.title("Change in loss")
plt.xlabel('Epoch')
plt.ylabel('Lost')
plt.show()


    
