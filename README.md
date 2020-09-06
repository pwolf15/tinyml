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