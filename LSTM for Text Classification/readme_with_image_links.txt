# Mini Task 1 Part 2

# RNN-Recurrent Neural Network

In RNNs we have data and the neural network learns over time and also holds the past information along with it so that it could be used again.

It basically uses recurrent learning patterns to make sure it saves the past information, have a “MEMORY” while also learning new information. It is similar to humans in this regard, when we read a text say a sentence we read each word and save all the data until the last word and we interpret the sentence only because of how we understand all the words. So we want the computer to do the same. It has multiple applications like in the case of Natural Language Processing uses such as chatbots or predicting the next word of a sentence like in our search engines.

RNN works by having a single hidden layer and providing the same weights and biases for every word that is scanned thus making them dependent on each other. Changing of weights and biases affect the whole system rather than just one hidden layer like in a regular neural network.

https://www.researchgate.net/figure/The-standard-RNN-and-unfolded-RNN_fig1_318332317

The diagram on the left shows the RNN in a compressed format and on the right it shows the expanded format.

x(t) is the input at that instant of time. h(t) is the hidden state for that instant of time and o(t) is the corresponding output. The self arrow in the diagram on the left shows the recurrent nature of the neural network.

So the hidden state gets its input from the previous hidden state as well as the user input(here the word at that time instant). So it takes weights and biases for both of them to find the corresponding hidden state at that moment of time. This is then passed through a tanh function to get the values between 0 and 1 to get a uniformity in the hidden states got and then an output is produced using a different set of weights and biases from this. This hidden state is thus used for the calculation of next hidden state giving it a memory. This is done and then back propagated to get the proper weights and biases but the back propagation here is more complex due to the RNN being a dependent neural network.

https://media.geeksforgeeks.org/wp-content/uploads/Screen-Shot-2018-08-23-at-3.27.11-PM.png
https://media.geeksforgeeks.org/wp-content/uploads/Screen-Shot-2018-08-23-at-3.28.27-PM.png
https://media.geeksforgeeks.org/wp-content/uploads/Screen-Shot-2018-08-23-at-3.29.30-PM.png
# LSTM

For small amounts of data, an RNN would suffice. But as the complexity/size increases it would be hard for the RNN to keep track even though it would still have the data. So if we want to get information from later parts of the text, earlier parts might not be used there/ be forgotten. This would pose an issue. This is why idea of LSTMs were created. LSTM is Long Short Term Memory.

What an LSTM does is, in addition to a hidden state, it also has a cell state which is expected to carry important words or information from the given state until it would need it to make a prediction.

For example if a sentence has 3 lines and first 2 lines are about a female and the 3rd line is about a male, and we are predicting the upcoming words at various parts of the sentence. When we do the prediction in the first 2 lines we have to make sure it is giving the output in the female gender if needed while the 3rd one has a change of subject so the female association should be removed and a male association should be created and the output should be in the male gender.

https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-SimpleRNN.png

The input is first put through the forget gate - a sigmoid function to bring it between 0 and 1 and check whether the information should be added or deleted from there. If the output through the forget gate is near 0 that means the previous cell state is erased and if it is near 1 it isn't erased from the cell state.

https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-f.png

Then we have an input layer which has 2 parts. The values we wish to update and values we wish to add to the cell state. The ones which we wish to update are passed through a sigmoid and candidates are passed through a tanh and the corresponding weights are applied on them to get the corresponding state values.

https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-i.png

https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-C.png

The final new cell state is got from the product of these 2 state values and the forget gate. This helps in preserving the long term memory of the data.

The output and the new hidden state(from the given input and previous hidden state) is arrived upon from these values.
https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-o.png
