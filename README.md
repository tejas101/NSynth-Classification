# NSynth-Classification
This codebase was the submission for Project 1 of my course Introduction to Machine Learning.
The NSynth dataset used can be found here : https://magenta.tensorflow.org/datasets/nsynth
# The problem statment 
For this project, we use the NSynth dataset created by researchers from the Magenta project at Google Brain (https://magenta.tensorflow.org/datasets/nsynth). Your task is to classify single note recordings of instruments into one of 10 instrument family names (e.g., pipe organ, jazz organ, and synth organ recordings are all from the family “organ” in the dataset).

There are 11 instrument families (e.g., mallet, organ, piano, strings, guitar, etc.), and 3 instrument source types (acoustic instrument sample, electronic instrument sample, or synthesized sample). The test set does not include any samples for the ‘synth lead’ class, so please remove these from the training and validation sets.

You will create at least two convolutional neural networks, with the goal of recognizing these instrument families with high accuracy. You are welcome to use built-ins for the Tensorflow or Keras libraries, and to use layers/nodes of whatever type you like (e.g., dropout layers, sigmoids, ReLU, tanh, etc.), or build your own.

Input Data (Recordings)
Data is available on the CS machines at:

/home/fac/cmh/tfrecord

Each recording is 4 seconds long; the last second is just the release of the note (e.g., after lifting a piano key). The sample rate is 16kHz (16,000 samples/second; 64,000 samples over the four seconds). To relate this back to images, a 256x256 greyscale image has 65,536 pixels (i.e., not a large image, but a lot of input nodes for a network). This is a rather large input layer, and you will probably want to find a way to reduce the input size. Please keep in mind that you need twice as many samples per second as any frequency you wish to represent. So, for example, to capture a 60Hz frequency (pitch) without noise, you need a minimum of 120Hz (120 samples per second).

The data is stored in a TensorFlow format, as well as JSON files + .wav files for each sample. The JSON and .wav files provide a more intuitive way to browse through the data; the TensorFlow data is in a database format designed for rapid lookup. 

Implementation
Use python and Keras/Tensorflow to create your network. Submit your code and weights with a README in project1.zip. You are welcome to use other Python libraries (e.g., Pandas, Matplotlib Scitkitlearn, etc.) for the project.

Visualizations you will Need to Create
  1)Learning curves for training and validation data
  2)Visualization of a 1-D audio waveform (sequence of values in [-1,1] representing the volume at each time)
  3)For each class, waveform for samples where the correct class probability is very high, or very low.
  4)For each class, waveforms for samples near the decision boundary, where the probability for the correct class is slightly higher/lower than the other classes.
  5)Confusion matrix, visualizing the frequency of correct classifications and mis-classifications
