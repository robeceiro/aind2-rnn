import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series,window_size):
    # containers for input/output pairs
    X = []
    y = []
    for i in range (0,len(series)-window_size-1): #Move one by one, upto the end of the series given my window size
        x = []
        for j in range (i,i+window_size): #Look at the window series, without moving i
            #Save the new element in a temp array
            x.append(series[j])
        #Store the temp array
        X.append(x)
        #Store the output
        y.append(series[i+window_size])
        
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)
    
    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(step_size, window_size):
    # given - fix random seed - so we can all reproduce the same results on our default time series
    np.random.seed(0)


    # TODO: build an RNN to perform regression on our time series input/output data

    #Create a sequential model
    model = Sequential()
    #Add a LSTM layer with 5 hidden units and the specified input shape
    model.add(LSTM(5, input_shape=(window_size, 1)))
    #Add a fully connected module with one unit
    model.add(Dense(1))

    # build model using keras documentation recommended optimizer initialization
    optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # compile the model
    model.compile(loss='mean_squared_error', optimizer=optimizer)


### TODO: list all unique characters in the text and remove any non-english ones
def clean_text(text):
    # find all unique characters in the text
    print("Before")
    all_chars = ''.join(set(text))
    print(all_chars)
    #List all the allowed chars
    allowed_chars = 'AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz!?.,;: '
    # remove as many non-english characters and character sequences as you can 
    for c in all_chars:
        #Replace if necessary
        if c not in allowed_chars:
            text = text.replace(c,' ')
    #Use a while for double and triple blank spaces
    while '  ' in text:
        # shorten any extra dead space created above
        text = text.replace('  ',' ')
    print("After")
    all_chars = ''.join(set(text))
    #Output the result
    print(all_chars)


### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text,window_size,step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    
    for i in range (0,len(text)-window_size-1): #Use the same logic as in the previous function
        #Take the step size into account, continue the iteration if step size indicates so
        if i%step_size > 0:
            continue
        #Now the temp array is an empty string
        x = ''
        for j in range (i,i+window_size):
            x = x+text[j] #Add a new char to the temp string
        #Save the new input
        inputs.append(x)
        #Save the new output
        outputs.append(text[i+window_size])
            
    return inputs,outputs
