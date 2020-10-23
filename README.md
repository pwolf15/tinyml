# TODO

* Training the on/off model
* Gathering your own data
* Deploy visual wake word to Arduino and Sparkfun Edge

# Chapter 4: Hello World
* Goal: use Sine wave to power on/off LED

Steps
1. Obtain a simple dataset.
2. Train a deep learning model.
3. Evaluate the model's performance.
4. Convert the model to run on-device.
5. Write code to perform on-device inference.
6. Build the code into a binary.
7. Deploy the binary to a microcontroller.

Google Colaboratory
-> online env for Jupyter notebooks
TensorFlow: tools for building, training, evaluating, and deploying machine learning models.
Keras: Tensorflow's high-level API that makes it easy to build and train deep learning networks
TensorFlow Lite: set of tools for deploying TensorFlow models to mobile and embeded devices
* training, evaluation, testing data
* regression: model that takes an input value and uses it to predict a numeric output value
* Keras: Sequential model: layer of neurons stacked on top of the next
* Relu
* Dense layer: fully connected; each input will be fed into every single one of it sneurons during inference
* activation = activation((input * weight) + bias)
* ReLU: rectified linear unit
* Since ReLU is nonlinear, it allows multiple layers of neurons to join forces and model complex nonlinear relationships, in which y value doesn't increase by same amount fo revery incrmeent of x.
* compile configures important arguments used in training and prepares model to be trained.
model.summary(): print some summary information about architecture
* size fo network (memory) mostly depends on number of parameters, meaning total number of weights and biases
* Keras fit(): performs training
* each run-through of dataset == 
* batch: pieces of data passed to network; outputs' correctness is measured in aggregate and network's weights and biases are adjusted accordingly
* batch_size: how many pieces of training data to feed into network before measuring its accruacy and updating its weights and biases. Suggestion: start with batch size of 16 or 32.
* Training metrics; loss, mae, val_loss
=> loss: output of loss fnction
=> mae: mean absolute error fo training data
=> val_loss, val_mae (validation results)
* goal: stop training when model is no longer improving, or training loss is less than validation loss (overfitting)
* if model poorly approximates, easy way to improve performance is add another layer

TensorFlow Lite Converter: converts models into space-efficient format for use on memory-constrained devices, and it can apply optimizations that further reduce the model sie and make it run faster on small devices
TensorFlow Lite Interpreter: runs an appropriatel converted TensorFlow Lite model using the most efficient operations for a given device

Converter: Keras model => Flatbuffer file
quantization: weights and biases stored as floats; quantization allows you to store as integers

Executing TensorFlow Lite models
1. Instantiate an interpreter
2. Call some methods that allocate memory for the model
3. Write the input to the input tensor.
4. Invoke the model.
5. Read the output from the output tensor.

computation graph: all logic that makes up the architecture of our deep learning network.

Size of model for sine wave ==> 2 Kb, 224 less for quantized version

https://colab.research.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/examples/hello_world/train/train_hello_world_model.ipynb#scrollTo=l4-WhtGpvb-E

Application code
- preprocessing: transforms inputs to be compatible with the model
- TF Lite interpreter: runs the model, model makes predictions on data
- postprocessing: interprets the model's output and makes decisions
- output handling: uses device capabilities to respond to prediction
xxd: translates model into C++
tensor arena: area of memory used to store model's input, output, and intermediate tensors
choosing right arena size requires trial and error
- TFLiteTensor struct, provides API for tensors
- Keras layers accept input as 2d tensor for scalar value
- TfLitePtrUnion for data on inputs
- For multi-dimensional data, add to input as flattened values
- model = graph of operations which the interpreter exxecutes to transform the input data into an output
- TF_LITE_MICRO_EXPECT_NEAR: use thresholding
- C++ prefix constants with a k


# Chapter 6

* Microcontroller is connected to the circuit board using pins. GPIO pins, since they are general purpose input/output
* GPIO pins are digital
* Some microcontrollers also have analog digital pins
* Arduino: Arduino Nano 33 BLE Sense
* Pulse-width modulation used to dim LED; switch an output pin on and off extremely rapidly; the pin's output voltage becomes a factor of the ratio between time spent in the off and on states
* Arduino often uses serial port for debugging information
* Serial plotter can display a grpah of values it receives via serial
* ErrorReporter outputs data via Aruino's serial interface
* Needed to install board support package
* Also needed to add my user to the dialout group
* Change animation: added back and forth animation by incrementing number and outputting symbol (+ or -) in serpentine fashion. Each line is 16 characters
* Change what you're driving: drove GPIO pin #2 (connected up to speaker and LED)

# Chapter 7
- Wake-word detection: words which trigger audio stream to speech detection model
- Wake-word detection is a part of cascading architecture, where a tiny, efficient model "wakes up" a larger, more resource-hungry model.
- 18 KB model classifies spoken audio
- model trained to recognize "yes" and "no"
- application will indicate that it has detected a word by lighting an LED or displaying data on a screen

## General TinyML Application architecture
1. Obtains an input
2. Preprocesses the input to extract features suitable to feed into a model
3. Runs inference on the processed input
4. Post processes the model's output to make sense of it
5. Uses the resulting information to make things happen.

## Wake-word application architecture complexiities
1. Audio data as input which requires heavy processing before input.
2. Model is a classfiier, outputting class probabilities. We'll need to parse and make sense of output.
3. Designed to perform inference continually, on live data. We'll need to write code to make sense of a stream of inferences.
4. Model is larger and more complex. We'll push the hardware to limits of its capabilities.

Speech Commands dataset
- yes, no, unknown, silent
- 65000 second long utterances of 30 short words
- model takes sepctrograms, 2D arrays that are made up of slices of frequency information, each taken from a different time window.
- convolutional neural network: works well with multidimensional tnesors in which information is contained in the relationships between groups of adjacent values
- images,
- Goal: display spectrogram
- components
* main loop
- audio provider: audio data capture via microphone
- remember you wan to minimize the number of ops you import to your device
- feature provider: converts raw audio data into spectrogram format
- TF Lite interpreter
- model: data array
- command recognizer: aggregate reults and determines on average whether a known word is heard
- command responder: uses device's output capabilities to let user know whether command was heard

Each spectrogram == 2D array, with 40 columns and 49 rows, where each rows represents a 30-ms sample of audio split into 43 frequency buckets. 30-ms audio into fast Fourier trnasform; analyzes frequency distribution of audio in the sample and creates an array of 256 frequency buckets, each with a value from 0 to 255. These are averaged together into groups of six, leaving us with 43 buckets. (micro_features_generator)

Command recognizer uses multiple inferences to determine whether command inputs cross a given threshold.

Sparkfun Edge Setup:
https://learn.sparkfun.com/tutorials/using-sparkfun-edge-board-with-ambiq-apollo3-sdk/all


Can I use wake word detection to hook up with Pynq and determine what I'm saying?

- https://stackoverflow.com/a/57435184
-> sparkfun edge wakeword generally having problems? Why?

## Training the wake word model

* Configuring parameters
* Installing the correct dependencies
* Monitoring training using something called TensorBoard
* Running the training script
* Converting the training output into a model we can use

* Training usually 5-10x faster with the GPU

* cross_entropy: models loss, which quantifies how far from the correct values the models predictions are.
* checkpoint files: file written by training script which contains the weights and biases produced during traning process.
* Tensorflow model contains two things
1. weights and biases from training
2. a graph of operations that combine the model's input with these weights and baises to produce to model's input

https://colab.research.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/examples/micro_speech/train/train_micro_speech_model.ipynb#scrollTo=-RHFQVLJnQTa

creating the model file => freezing: create a static representation of the graph with weights frozen into it
Still a TensorFlow model to TensorFlow lite model conversion process; converter called toco
=> Doesn't apply to example above, skips to the C file generation step

Feature generation is main step for pre-processing input. The reasons for doing so is that the resulting data is easier to classify/differentiate. Additionally, the data size is much smaller (1960 vs 16000 values). Most models operate on processed data, but some models like DeepMind's WaveNet operate on raw data.

How does feature generation work?
- mel-frequency cepstral coefficients (MFCC)
- different approach here; used by google, but not published
1. Fast Fourier Transform for a given time slice (in our case 30 ms); data filtered with Hann window. Hann window reduces influence of samples at either end of the window.
2. 256 frqeuency buckets
3. Scale down by non-linear Mel function so low frequencies have more resolution

See Trainable Frontend for Robust and Far-Field Keyword sSpotting from Yuxuan Wang et al
There is a step in background noise reduction. Even and odd buckets have different coeffcients.
Per-channel amplitude normalization occurs after this to boost the signal based on the running average noise.

Convolutional layers are used for spotting 2D patterns in input images.
Think of convolution as moving a series of rectangular filters across the iamge, with the result at each pixel for each filter corresponding to how similar the filter is to the that patch in th eimage.
Fully connected layer: weight for every value in the input tensor. Result is indication of how closely the input matches the weights, after comparing every value. Each class has its own weights, so there's an ideal pattern for "silence", "unknown", "yes", and "no"
Softmax layer: effectively helps increase the difference between the highest output and its nearest competitors
There is a ReLU activation function after eah layer, in addition to biases. Again, ReLU helps results converge much moreq uickly.
Model runs 10 to 15 seconds (empirical choice), and then the score is averaged across time. 

More sophisticated speech recognition algorithms accept a stream of input and use recurrent neural networks instead of single-layer convlutional neural networks as done here. Having streaming baked into the model means that you don't need to do the postprocessing done here.

## Training your own model

* need to gather thousands of audio samples in most cases
* Open Speech recording app was used for Speech Commands dataset

# Chapter 9: Person Detection

* simpler than audio
* uses CNN since they work well with adjacent data, multidimensional tensors
* model uses raw pixel data, unlike audio (need to translate raw data into spectrogram)
* Uses Visual Wake Words dataset
* 250 KB model
* 96 x 96 x 1 grayscale images as input
* state-of-the-art image classifiers often work with 320 x 320
* MobileNet architecture: well-known and battle-tested architecture for image classification on devices like mobile phones
* inference takes longer due to size of input => one inference per several seconds, vs. multiple inferences per second
* Keep in mind an output tensor might have more dimensions than expected; this can be due to the architecture or due to implementation. 

## Chapter 10: Training Wake Word Model

* Training this model takes several days, 100s GBs of storage