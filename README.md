# machine-learning-2021

01/06/20
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
