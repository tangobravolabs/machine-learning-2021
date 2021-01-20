# machine-learning-2021


01.20.21
TIL:https://en.wikipedia.org/wiki/Gradient_descent
* An analogy for understanding gradient descent
> Fog in the mountains
> The basic intuition behind gradient descent can be illustrated by a hypothetical scenario. A person is stuck in the mountains and is trying to get down (i.e. > > > trying to find the global minimum). There is heavy fog such that visibility is extremely low. Therefore, the path down the mountain is not visible, so they must > use local information to find the minimum. They can use the method of gradient descent, which involves looking at the steepness of the hill at their current > > > position, then proceeding in the direction with the steepest descent (i.e. downhill). If they were trying to find the top of the mountain (i.e. the maximum), > > > then they would proceed in the direction of steepest ascent (i.e. uphill). Using this method, they would eventually find their way down the mountain or possibly > get stuck in some hole (i.e. local minimum or saddle point), like a mountain lake. However, assume also that the steepness of the hill is not immediately obvious > with simple observation, but rather it requires a sophisticated instrument to measure, which the person happens to have at the moment. It takes quite some time > > to measure the steepness of the hill with the instrument, thus they should minimize their use of the instrument if they wanted to get down the mountain before >  > sunset. The difficulty then is choosing the frequency at which they should measure the steepness of the hill so not to go off track.

In this analogy, the person represents the algorithm, and the path taken down the mountain represents the sequence of parameter settings that the algorithm will explore. The steepness of the hill represents the slope of the error surface at that point. The instrument used to measure steepness is differentiation (the slope of the error surface can be calculated by taking the derivative of the squared error function at that point). The direction they choose to travel in aligns with the gradient of the error surface at that point. The amount of time they travel before taking another measurement is the step size.

01.19.21
TIL: https://medium.com/towards-artificial-intelligence/how-i-developed-a-game-using-computer-vision-18409a39a1f3
* Read the tutorial to create a game using CV
* There's a target object, in this case a car
* There's a game play area (road with objects)
* By moving the car, left and right you can move the on-screen car to avoid obstacles
* The author used a red car because it is easy to isolate from the background
* I also forked a branch for a conmplete YOLO3 model. Tried running it but got loading errors and a strange GPU warning.


01.18.21
TIL: https://queirozf.com/entries/pandas-dataframe-plot-examples-with-matplotlib-pyplot
* Differnt ways to visualze the data
Also,:https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html
* Getting descriptive summaries of the data
* And selecting specific values 

01.17.21
TIL:https://www.kdnuggets.com/2020/08/5-different-ways-load-data-python.html
* Having trouble running notebooks because I skipped the Python part about loading and manipulating dat
* Here are 5 ways to load the data
Also: https://www.geeksforgeeks.org/ways-to-import-csv-files-in-google-colab/
* Specifically how to load CSV files and get the data in a state that it usable
* Bonus: https://ourcodingclub.github.io/tutorials/pandas-python-intro/
* Super helpful Pandas overivew
* The series is a one-dimensional array-like structure designed to hold a single array (or ‘column’) of data and an associated array of data labels, called an index
* The DataFrame represents tabular data, a bit like a spreadsheet. DataFrames are organised into colums (each of which is a Series), and each column can store a single data-type, such as floating point numbers, strings, boolean values etc. DataFrames can be indexed by either their row or column names. (They are similar in many ways to R’s data.frame.)
* Dictionaries are a core Python data structure that contain a set of key:value pairs. If you imagine having a written language dictionary, say for English-Hungarian, and you wanted to know the Hungarian word for “spaceship”, you would look-up the English word (the dictionary key in Python) and the dictionary would give you the Hungarian translation (the dictionary value in Python). So the “key-value pair” would be 'spaceship': 'űrhajó'.
* Note that Pandas DataFrames are accessed primarily by columns. In a sense the row is less important to a DataFrame. For example, what do you think would happen using the following code?
* Also: https://www.youtube.com/watch?v=cN7W2EPM-dw
* Colab get from GoogleSheets



01.16.21
TIL: https://www.kaggle.com/dansbecker/building-models-from-convolutions
* When the tensors are applied to the image, each filtered setion has an output - those output sections are compiled in a new output tensor
* There can be multiple filters applied which results in a layers (or channels) of output tensors
* You can move left-to-right, top-to-bottom through a channel or vertifcally through through the layers (the channel dimension)
* As you get more layers, you can get more complex patterns - concentric circles might be a tire, concentric circles togther might be a car
* You can get predictions from these recognized patterns
* Many breakthroughs in Computer Vision have come through a competition called, "Imagenet"
* Many of these models can accuratley detect the subject about 80% of the time - first time. This is better than humans.
* You often start with one of these pre-trained models to detect object because they have so many layer / patterns already built in.
* CV learning process - start wtih a pre-trained model
* Get it working for yourself
* Use transfer learning to build on that model to meet your needs for a new, custom detection model
Also: https://www.kaggle.com/dansbecker/tensorflow-programming
* > + Gather your images and put them in a directory
* > + Access the images via your notebook
* > + Preprocess the images (i.e sizing is important here - only use the size image you need)
* > + Convert each image into an array using an image-to-array function (this turns the image into an array.) Adding additional images create a 4th dimention (the tensor group for each image, then pass through images)
* > + Preprocess the pixel values so that they are between -1 and 1 (this is what them model is expecting, so we need to convert the image) 
* > + Then, specify the model and weights
* > + Model = ResNet50 in this example
* > + This model has 1,000 trained labeled patterns, or objects it could detect, so we'll limit the output to the top 3
* > + ARGHHH!! I keep getting stuck at the filedir / docker misalignment.


01.15.21
TIL: https://www.kaggle.com/dansbecker/intro-to-dl-for-computer-vision
* Tensorflow is the leading tool for deep learning
* Keras is a API-baed way to get data into and out of models
* Tensorflow is so popular that Keras built a library inside of Tensorflow
* Tensor is the word for something like a matrix but can have any number of dimensions
* A convulusion is a small tensor that can be multiplied over the main image [data set] (like a filter) to find patterns
* If you multiply the convulusion over a section of the image that aligns with the pattern, you get a large value
* If you multiply the convulusion over a section of the image that does not align well with the pattern, you get a smaller value
Also: https://www.kaggle.com/tangobravo2/exercise-intro-to-dl-for-computer-vision/edit
* The deep learning technique determines what convolutions will be useful from the data (as part of model-training)
* The convolutions you've seen are 2x2. But you could have larger convolutions. They could be 3x3, 4x4, etc. They don't even have to be square. Nothing prevents using a 4x7 convolution.


01.14.21
Read: https://medium.com/cracking-the-data-science-interview/how-to-think-like-a-data-scientist-in-12-steps-157ea8ad5da8
* Summary of the book: https://www.manning.com/books/think-like-a-data-scientist
* The 1st phase is preparation — time and effort spent gathering information at the beginning of a project can spare big headaches later
* > Setting goals : Every project has a customer. 
* > Exploring data : https://miro.medium.com/max/347/1*3i9lB6RPCuplb8yyaJODwQ.jpeg
* > Wrangle the data
* > Assessing the data
* The 2nd phase is building the product, from planning through execution, using what you learned during the preparation phase and all the tools that statistics and software can provide.
* > Plan
* > Engineer
* > Optimize
* > Execute
* The 3rd and final phase is finishing — delivering the product, getting feedback, making revisions, supporting the product, and wrapping up the project.
* > Deliver
* > Revise
* > Wrap-up
* Nice visual of all the phases: https://miro.medium.com/max/500/1*r2s0u-zlUyxYjIYPR62xrA.jpeg


01.13.21
pp.134-137
TIL:
* How to run clustering
* How to visualize clustering
* How to select the optimal # of clusters from the chart
* Run a summary table of clusters

01.12.21
pp. 132-133
TIL:
* get a dataset from a URL and load into Pandas
* read from a delimited CSV file
* Load dython to visuaize the data
* Build a 16x16 association matrix



01.11.21
TIL:https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.line.html



01.10.21
TIL: https://colab.research.google.com/notebooks/intro.ipynb#scrollTo=-Rh3-Vt9Nev9
* How to connect colab to my google drive
* Create a txt file in python and store in my drive
* Open and read a txt file in my notebook
* Create a google sheet in python
* Add content in python to my google sheet
* view the data in my notebook via pandas

01.09.21
pp: 30-39
TIL:
* How to visualize the data using seaborn
* How to create a model from a data set in scikit
* How to split data in to training and validation data sets in scikit
* How to check the performance of model in scikit
* How to print out a confusion matrix
* What a clear confusion matrix should look like
* How to create a model in Keras

01/08/21
pp: 20-30 (Cookbook)
TIL:
* How to load packages
* How to execute an extension
* How to calculate runtime
* How to show a progress bar for runtime
* How to compile code
* Simple Data Solution Workflow
> + Load the dataset
> + Visualize the data
> + Preprocess and transform the data
> + Choose a model to use
> + Check the performance
> + Interpret and undertand the model


01/07/21 
Read: https://towardsdatascience.com/the-not-so-naive-bayes-b795eaa0f69b
* Naive Bayes just uses Bayes underlying theorom as a construct for evaluation
* Always the first step is gathering data!
* This spam filter used most frequently used words to create distinct models of what non-spam and spam messages ar
* Transforming or pre-processing the data by removing "stop words" like a, the, and... etc.
* Training the model requires train and test data (general rule of thumb) is 80% train, 20% test
* Models must evolve - rarely is the first iteration of training enough (and should always be verified, if you think it is.)
* As you learn more about the data, you can evole your ways of processing the data and approaches to modelling to drive better results
* Bayes is a great place to start to test hypotheses and start to mdoel the data so you can move to your next step
Read: https://stats.stackexchange.com/questions/246101/when-to-use-bernoulli-naive-bayes/246106
* Bernoulli Naive Bayes is for binary features only. Similarly, multinomial naive Bayes treats features as event probabilities.


01/06/21
Watched: https://www.youtube.com/watch?v=inN8seMm7UI
TIL:
* Google Colab is like Jupyter notebooks hosted in Google Drive https://colab.research.google.com/
* You can write Python code in cells in Google Colab
* * It also has a built in code snippet library
* The files can be shared with others standard Jupyter files
* You an import Jupyter files from others
* pip installed:
> * tqdm for progress bars
> * swifter for quicker pandas processing
> * ray and joblib for multiprocessing
> * numba for JIT compilation
> * jax for array processing with autograd
> * cython for comiling Cython extentions in the notebook
> * seaborn for data visualization 
Read: https://towardsdatascience.com/getting-started-with-google-colab-f2fff97f594c
Read: https://medium.com/towards-artificial-intelligence/google-colab-101-tutorial-with-python-tips-tricks-and-faq-7689bd4d24b4
* Found Kaggle https://www.kaggle.com/
> Access free GPUs and a huge repository of community published data & code.

01/05/21
Read: https://medium.com/swlh/applications-of-machine-learning-9f77e6eb7acc
TIL:
* The intelligent systems built on machine learning algorithms have the capability to learn from past experience or historical data.
* Machine learning is very useful in any application that’s based on pattern recognition
* Pattern Heirarchy
> * Problem domain (ex. Speech Recognition)
> * Application (ex. inquiry with automated operator)
> * Input Pattern (Spoken waveform)
> * Pattern Class (Spoken words)
* Pipeline = IRL activity / object > digital representation > signal processing > pattern definition (training) /recognition (production)
* Sources:
> * Sensory detection: visual, audio, temperture, light, humidity, lidar, infrared, vibration, motion, etc.
> * Data files/sources: images, structured data, unstrucutred data, social media, dark data (logs, events, ets)
* Dark Data: https://www.kdnuggets.com/2015/11/importance-dark-data-big-data-world.html
* Dark data is a subset of big data but it constitutes the biggest portion of the total volume of big data collected by organizations in a year.
* A large portion of the collected data are never even analysed.
* It is surprising because at the time of data collection, the companies assume that the data is going to provide value. 
* Why is it dark?
> * Disconnect among departments
> * Technology and tool constraints
* Problems dark data can cause
> * Legal and regulatory issues
> * Intelligence risk
> * Loss of reputation
> * Opportunity costs


01/04/21
Working from https://www.amazon.com/Artificial-Intelligence-Python-Cookbook-algorithms/dp/1789133963
pp. 9-16
TIL:
* Google Colab is Jupyter Notebooks in the cloud
> Install Jupyter Notebooks: https://jupyter.org/install and the docs: https://jupyterlab.readthedocs.io/en/stable/
> * This is good for doing work on your computer. Google Colab has a 12-hour timeout.
> Access Google Colab https://colab.research.google.com/


01/03/21
Read: https://atharvaaingle.medium.com/10-tips-to-learn-machine-learning-72b7dcf15528
TIL:
* Both Python and R are important
* Tutorials allow you to recreate, but you need to apply to your own data / projects to get real learning application benefits
* Lab / hacking results aren't enough, you need to deploy in real-world situations to understand how your model is performing
* How to make algorithmns in excel: https://www.quora.com/How-do-I-create-an-algorithm-with-a-spreadsheet
> * Once an Excel formula has more than about three levels of nested parentheses, it usually gets pretty messy. Whenever you have a moderately complex algorithm, try to break it into logical steps. Perhaps you could put raw data in Columns A and B. Do a calculation in Column C. Put the results of an “if” statement in Column D. If “D” is true, perform another calculation in Column E. Otherwise, perform an alternative calculation in Column F.
> *Column D will either be True or else False. If you see a value that is true when you know it should be false, there is an immediate source of feedback for you to trace the source of inaccuracies and inconsistencies.
* Due to the abstract nature of data science, feedback or checking your answers is super important.
> How to create a dependent dropdown in excel: https://productivityspot.com/dependent-drop-list-google-sheets/
* First attempt at an estimating model: https://docs.google.com/spreadsheets/d/1lZz_DFdhKHz3KqRUN2olbjs0t4-Z33xVpVika5UuH48/edit?usp=sharing


01/02/21
Read: https://jaswanth-badvelu.medium.com/the-quickest-way-to-build-dashboards-for-machine-learning-models-ec769825070d
TIL: 
* While building complex machine learning models is difficult, conveying the predictions of trained machine learning models to stake holder’s with no technical background is even more cumbersome.
* Explainerdashboard is a simple python tool to create dashboards to explain the models using pythong
> Explainerdashboard documentation: https://explainerdashboard.readthedocs.io/en/latest/explainers.html
* Random forest: https://en.wikipedia.org/wiki/Random_forest
> *  A lot of decision trees can ensure the accuracy of the random forest
> *  Random Forest is more accurate than decision trees. Decision trees are easy to trace to outcomes. Random forest is not. Used in conjunction, the random forest can comnfirm decision tree accuracy and outcomes.
* Random forest vectors are like kernels: https://en.wikipedia.org/wiki/Kernel_method
> * Kernel machines are a class of algorithms for pattern analysis
> * Kernel machines use kernel functions, which enable them to operate in a high-dimensional, implicit feature space without ever computing the coordinates of the data in that space, but rather by simply computing the inner products between the images of all pairs of data in the feature space. 
> * AKA the Kernel trick "replacing its features (predictors) by a kernel function."
> * Any linear model can be turned into a non-linear model by applying the kernel trick to the model: replacing its features (predictors) by a kernel function.
> * The kernel trick avoids the explicit mapping that is needed to get linear learning algorithms to learn a nonlinear function or decision boundary. 
* Decision boundary
> * In a statistical-classification problem with two classes, a decision boundary or decision surface is a hypersurface that partitions the underlying vector space into two sets, one for each class. The classifier will classify all the points on one side of the decision boundary as belonging to one class and all those on the other side as belonging to the other class.
> * A decision boundary is the region of a problem space in which the output label of a classifier is ambiguous.

Q: How does decision boudary relate to project scope?
-- In terms of project implementation, the boundaries of a project are the reasonable limits of project work to determine what is included in the project and what’s not. The boundaries are defined as measurable and auditable characteristics and closely linked to project objectives. They create a holistic project perception, determine limits and exclusions of the project, and form the content of project scope in terms of expected results.
-- ** It reduces supervision and the need for control while ensuring higher project performance.**



01/01/21
Read: https://nayakvinayak95.medium.com/author-identification-using-naive-bayes-algorithm-1-abeeb88eb862
TIL:
* How to scrape web data and a GitHub repo with a scraping tool
* Reinforced the pipeline steps of:
> 1) Identify a problem
> 2) Collect data
> * Note: You need training data and test data
> 3) Process data
* Naive Bayes is a good starting place for a classification problem
* Comparing train data and test data, you want to have a similar distribution
* Bulleted lists use an asterisk ( * ), plus sign ( + ), or hyphen ( - ) to delimit each item. There is at least one space between the delimiter and the item. The delimiter has no effect on the marker shown in the rendered text
 > https://daringfireball.net/projects/markdown/syntax#blockquote

Q:
* Is there a tool or method to understanding what size data set is needed to do training and test for a specific problem?
-- https://datascience.stackexchange.com/questions/19980/how-much-data-are-sufficient-to-train-my-machine-learning-model
-- A general rule (there are others) = 10X the number of features
