# machine-learning-2021

10.27.21
TIL: https://machinelearningmastery.com/basic-data-cleaning-for-machine-learning/
* In tabular data, there are many different statistical analysis and data visualization techniques you can use to explore your data in order to identify data cleaning operations you may want to perform.
* There are some very basic data cleaning operations that you probably should perform on every single machine learning project.
> * Identify and remove column variables that only have a single value.
> * Identify and consider column variables with very few unique values.
> * Identify and remove rows that contain duplicate observations.
* Data cleaning refers to identifying and correcting errors in the dataset that may negatively impact a predictive model.
* The so-called “oil spill” and "iris" datasets are a standard machine learning dataset.
* > https://raw.githubusercontent.com/jbrownlee/Datasets/master/oil-spill.csv
* > https://raw.githubusercontent.com/jbrownlee/Datasets/master/oil-spill.names
* > https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv
* > https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.names
* When a predictor contains a single value, we call this a zero-variance predictor because there truly is no variation displayed by the predictor.
* You You can detect rows that have that have a single value using the unique() NumPy function that will report the number of unique values in each column.
* A simpler approach is to use the nunique() Pandas function that does the hard work for you.
* Another approach to the problem of removing columns with few unique values is to consider the variance of the column.
* Rows that have identical data are probably useless, if not dangerously misleading during model evaluation.
* Data deduplication, also known as duplicate detection, record linkage, record matching, or entity resolution, refers to the process of identifying tuples in one or more relations that refer to the same real-world entity.
* The pandas function duplicated() will report whether a given row is duplicated or not. All rows are marked as either False to indicate that it is not a duplicate or True to indicate that it is a duplicate.
* There are many ways to achieve this, although Pandas provides the drop_duplicates() function that achieves exactly this.
* 

10/26/21
TIl: https://towardsdatascience.com/understanding-and-implementing-dropout-in-tensorflow-and-keras-a8a3a02c1bfa
* Dropout is a common regularization technique that is leveraged within state-of-the-art solutions to computer vision tasks such as pose estimation, object detection or semantic segmentation.
* Neural networks have hidden layers in between their input and output layers, these hidden layers have neurons embedded within them, and it’s the weights within the neurons along with the interconnection between neurons is what enables the neural network system to simulate the process of what resembles learning.
* The disadvantage of utilizing deeper neural networks is that they are highly prone to overfitting.
* The primary purpose of dropout is to minimize the effect of overfitting within a trained network.
* Dropout technique works by randomly reducing the number of interconnecting neurons within a neural network. At every training step, each neuron has a chance of being left out, or rather, dropped out of the collated contribution from connected neurons.
* Also: https://www.seebo.com/machine-learning-ai-manufacturing/
* Artificial Intelligence is not a silver bullet; no solution will solve all, or most, of your problems. As a rule of thumb, AI works best when it is applied to solving a specific problem, or a very closely-related set of problems. 
* The two major use cases of Machine Learning in manufacturing are Predictive Quality & Yield, and Predictive Maintenance.
* Instead of performing maintenance according to a predetermined schedule, or using SCADA systems set up with human-coded thresholds, alert rules and configurations, predictive maintenance uses algorithms to predict the next failure of a component/machine/system.
* Predictive Quality and Yield automatically identifies the root causes of process-driven production losses using continuous, multivariate analysis, powered by Machine Learning algorithms that are uniquely trained to intimately understand each individual production process.
* Automated recommendations and alerts can then be generated to inform production teams and process engineers of an imminent problem, and seamlessly share important knowledge on how to prevent the losses before they happen.


01.25.21
TIL: https://aws.amazon.com/rekognition/?nc=sn&loc=1&blog-cards.sort-by=item.additionalFields.createdDate&blog-cards.sort-order=desc
* Amazon Rekognition makes it easy to add image and video analysis to your applications using proven, highly scalable, deep learning technology that requires no machine learning expertise to use.
* With Amazon Rekognition Custom Labels, you can identify the objects and scenes in images that are specific to your business needs. 
* You can quickly identify well known people in your video and image libraries to catalog footage and photos for marketing, advertising, and media industry use cases. 
* PPE detection -Personal Protective Equipment (PPE) detection
Also, https://aws.amazon.com/textract/
* Use textract to build automated document worfklows


01.24.21
TIL: https://towardsdatascience.com/understanding-the-basics-of-data-science-883b0ebbf5e0
* What does a ‘relevant’ dataset looks like?
* In the examples above, the left table dataset columns are totally not related to each other. The patient identifier has nothing to do with the current stock price and the premium amount of different insurance plans.
* Whereas, in the table on right, each row can be associated with a particular patient. The different columns are actually qualifying the patient and are termed as ‘Features’ of the dataset.
* Estimating the right amount of data needed is crucial while solving a data science problem. The rule of thumb is for e.g. if you have say 20 categorical variables with 15 levels / classes each and 18 continuous variables (for continuous variables assume 10 levels / classes) then the minimum number of rows needed is: (20 * 15 + 18 * 10) * 100 = 48,000 rows. (100 is just a multiplying factor and can be varied depending on how much data is available and the application area under consideration)
* If the dataset has a lot of missing values then the data is called a sparse data.
* Key data science problems (types)
* How much / how many?
* > Linear Regression, Polynomial Regression
* Which category?
* > Classification algorithms enable us to answer questions in which we have to predict a category. Classification algorithms are those which throw a category when you know that you have to bucket each entity into fixed number of classes or categories. These are also called as ‘Supervised Classifiers’.
* Which group?
* > So what if you don’t know the number of categories that the data needs to be divided into. Don’t worry, there are algorithms to serve you for this purpose as well. Such algorithms are called ‘Unsupervised Algorithms’: Clustering, Recommendation Engine.
* Is the behavior not normal?
* > This kind of a question might seem vague in the beginning but it has a lot of practical examples. The algorithms are referred to as ‘Anomaly Detectors’.
* What action?
* > Now in the last category, there are some systems in which the decisions need to be made in real time and there is no pile of data to work with but there is a flow of data to work with. Moreover, change in the environment variables changes the whole set of rules that were applied in the earlier scenario. Learning agents learn by interacting with its environment and observing the results of these interactions. This mimics the fundamental way in which humans (and animals alike) learn.

01.23.21 TIL: https://towardsdatascience.com/machine-learning-basics-descision-tree-from-scratch-part-i-4251bfa1b45c
* Regression-type problems are generally those where we attempt to predict the values of a continuous variable from one or more continuous and/or categorical predictor variables. 
* If we used simple multiple regression, or some general linear model (GLM) to predict the selling prices of single-family homes, we would determine a linear equation for these variables that can be used to compute predicted selling prices.
* Classification-type problems are generally those where we attempt to predict values of a categorical dependent variable (class, group membership, etc.) from one or more continuous and/or categorical predictor variables
* For example, we may be interested in predicting who will or will not graduate from college, or who will or will not renew a subscription.
* In other cases, we might be interested in predicting which one of multiple different alternative consumer products (e.g., makes of cars) a person decides to purchase, or which type of failure occurs with different types of engines. In those cases, there are multiple categories or classes for the categorical dependent variable.
* Attribute selection measure is a heuristic for selecting the splitting criterion that partition data into the best possible manner. It is also known as splitting rules because it helps us to determine breakpoints for tuples on a given node
* Information gain is a statistical property that measures how well a given attribute separates the training examples according to their target classification
* We can build a conclusion that less impure node requires less information to describe it. And, the more impure node requires more information.
* The more “impure” a dataset, the higher the entropy and the less “impure” a dataset, the lower the entropy
* Note that entropy is 0 if all the members of S belong to the same class. For example, if all members are positive, Entropy(S) = 0. Entropy is 1 when the sample contains an equal number of positive and negative examples. If the sample contains an unequal number of positive and negative examples, entropy is between 0 and 1.
* Information Gain = Entropy(parent node) — [Avg Entropy(children)]
* Gini says, if we select two items from a population at random then they must be of same class and probability for this is 1 if population is pure.
* Higher the value of Gini higher the homogeneity.
* CART (Classification and Regression Tree) uses the Gini method to create binary splits.
* Decision tree tutorial: https://www.hackerearth.com/practice/machine-learning/machine-learning-algorithms/ml-decision-tree/tutorial/
Also, https://brainhub.eu/blog/how-to-approach-data-science-problem/
* “Data science and statistics are not magic. They won’t magically fix all of a company’s problems. However, they are useful tools to help companies make more accurate decisions and automate repetitive work and choices that teams need to make,” writes Seattle Data Guy, a data-driven consulting agency.
* Some common problems that are solved with data science
> * Identifying themes in large data sets: Which server in my server farm needs maintenance the most?
> * Identifying anomalies in large data sets: Is this combination of purchases different from what this customer has ordered in the past?
> * Predicting the likelihood of something happening: How likely is this user to click on my video?
> * Showing how things are connected to one another: What is the topic of this online article?
> * Categorizing individual data points: Is this an image of a cat or a mouse?
Common approaches to solving these problems
> * Two-class classification: useful for any question that has just two possible answers.
> * Multi-class classification: answers a question that has multiple possible answers.
> * Anomaly detection: identifies data points that are not normal.
> * Regression: gives a real-valued answer and is useful when looking for a number instead of a class or category.
> * Multi-class classification as regression: useful for questions that occur as rankings or comparisons.
> * Two-class classification as regression: useful for binary classification problems that can also be reformulated as regression.
> * Clustering: answer questions about how data is organized by seeking to separate out a data set into intuitive chunks.
> * Dimensionality reduction: reduces the number of random variables under consideration by obtaining a set of principal variables.
> * Reinforcement learning algorithms: focus on taking action in an environment so as to maximize some notion of cumulative reward.
* Source of tools to solve problems: https://paperswithcode.com/



01.22.21
TIL:https://github.com/LianShuaiLongDUT/Keras-Res50-classfication/blob/master/main.py
And: https://keras.io/api/applications/resnet/
* Able to use Josh's notebook to run idenfication on toilets and sinks
* Able to modify the image used to run toilet and syncs on any image (from a URL)
* Able to modify the notebook to look for other items in the coco data set
> * https://cocodataset.org/#explore
* Able to change the theshold and see the results
* Able to see the output when "nothing" is identified (confidence too low)
* Able to see the output when "too many things" are identified (confused)
* Also, https://towardsdatascience.com/deploy-a-machine-learning-model-from-a-jupyter-notebook-9257ae5a5f7c
* How to get from a notebook to a machnine learning model
* This example uses IBM Watson to do most of the heavy-lfting
* Here's how to do it in AWS (requires SageMaker) https://aws.amazon.com/getting-started/hands-on/build-train-deploy-machine-learning-model-sagemaker/
* Also, https://valohai.com/blog/leveling-up-your-ml-code-from-notebook/
* Some info on why you shouldn't deploy a production system with a notebook
* PyCharm supports a science mode which allows you to run notebooks so you can transition to code
* Also, https://medium.com/@_orcaman/jupyter-notebook-is-the-cancer-of-ml-engineering-70b98685ee71
* Harsh article on the drawbacks of notebooks
* Good section on the issues I have seen with the df (dataframe) variable and how it strangely works
* I think notebooks are a great learning tool, its like being able to sketch.
* It also gives the users a quick way to get up and running without knowing extensive amounts of code and too strict of syntax
* In line commenting and narrative blocks make the code way more transferrable (this is something that should be considered in regular code, imo
* Lastly, notebooks are shareable in a way that code is not.


01.21.21
TIL: https://blog.dataiku.com/make-data-prep-less-of-a-hassle
* Data outputs
> + **Data** Here, data is transformed and the output is more data (such as when an analyst sends a clean Excel file to a boss to review).
> + **Reports:** Think of the same analyst, who sends their boss a bar chart of last quarter’s performance.
> + **Models:** Data is used to build algorithmic models to help the organization make future predictions and improve business outcomes.
* Also, https://earthdata.nasa.gov/earth-observation-data/data-recipes
* Data recipes are tutorials or step-by-step instructions that have been developed by the Earth Observing System Data and Information System (EOSDIS) Distributed Active Archive Centers (DAACs) staff or EOSDIS systems engineers to help users learn how to discover, access, subset, visualize and use our data, information, tools and services. 
* A storymap is an interesting way to present data with narrative in a story format ex. https://storymaps.arcgis.com/stories/9ebbe1b54dc847f2a7dd01917c9f3071
* More on ArcGIS: https://storymaps.arcgis.com/

01.20.21
TIL:https://en.wikipedia.org/wiki/Gradient_descent
* An analogy for understanding gradient descent
> **Fog in the mountains**
> The basic intuition behind gradient descent can be illustrated by a hypothetical scenario. A person is stuck in the mountains and is trying to get down (i.e. > > > trying to find the global minimum). There is heavy fog such that visibility is extremely low. Therefore, the path down the mountain is not visible, so they must > use local information to find the minimum. They can use the method of gradient descent, which involves looking at the steepness of the hill at their current > > > position, then proceeding in the direction with the steepest descent (i.e. downhill). If they were trying to find the top of the mountain (i.e. the maximum), > > > then they would proceed in the direction of steepest ascent (i.e. uphill). Using this method, they would eventually find their way down the mountain or possibly > get stuck in some hole (i.e. local minimum or saddle point), like a mountain lake. However, assume also that the steepness of the hill is not immediately obvious > with simple observation, but rather it requires a sophisticated instrument to measure, which the person happens to have at the moment. It takes quite some time > > to measure the steepness of the hill with the instrument, thus they should minimize their use of the instrument if they wanted to get down the mountain before >  > sunset. The difficulty then is choosing the frequency at which they should measure the steepness of the hill so not to go off track.

In this analogy, the person represents the algorithm, and the path taken down the mountain represents the sequence of parameter settings that the algorithm will explore. The steepness of the hill represents the slope of the error surface at that point. The instrument used to measure steepness is differentiation (the slope of the error surface can be calculated by taking the derivative of the squared error function at that point). The direction they choose to travel in aligns with the gradient of the error surface at that point. The amount of time they travel before taking another measurement is the step size.

Also, bolding in Markdown: https://www.markdownguide.org/basic-syntax/
* To bold text, add two asterisks or underscores before and after a word or phrase. 
* To bold the middle of a word for emphasis, add two asterisks without spaces around the letters.

Also, the Rosenbrock function: https://www.sfu.ca/~ssurjano/rosen.html
* The Rosenbrock function, also referred to as the Valley or Banana function, is a popular test problem for gradient-based optimization algorithms

Data democratization: 
> * Data democratization is the ability for information in a digital format to be accessible to the average end user. The goal of data democratization is to allow non-specialists to be able to gather and analyze data without requiring outside help.  - Margaret Rouse, TechTarget.com
* https://www.littlemissdata.com/blog/data-democracy
* I’ve started to notice another less obvious hurdle; the ability to break down a business problem into a data problem.  
* We need to lower the barriers to access standard, non-sensitive business data.
* How does a non-data expert gather and analyze the data in a self-serve manner? 
* Dashboards are great at presenting answers to the "What" questions. 
* Understanding the why is almost always a more complicated question than a set of pre-canned dashboards can answer.  It typically involves a data deep dive that needs to be tackled from a variety of angles which have not been planned for. 
* To answer the "Why", we now understand the need for two things.  
+ A custom investigation that the pre-canned dashboards likely didn't account for. 
+ Participation by subject matter experts (SMEs) in the data investigation.  Better, would be if the SMEs could explore in a self serve manner.  
* Facilitated data analysis is the type of analysis that is still performed by the data analyst team, but the investigation is lead by the SME. 
* Self-serve data analysis is where the non-data expert SME could dive into the data to their hearts content.
* Cornerstones
+ Simple tools
+ Simple data
+ Quality enablement
+ Supportive enviornment

* https://segment.com/
* 

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
