Download Link: https://assignmentchef.com/product/solved-machine-learning-hw5
<br>
<h1></h1>

<ol>

 <li><strong> VC-dimension of Neural Networks – Upper bound. </strong>We will now finish what we have started in the previous recitation and assignment. Let C be the class of hypotheses implementable by neural networks (NN) with <em>L </em>layers (including the output layer, excluding the input layer), each layer has exactly <em>d </em>nodes (except the output layer which has a single node), and the sign activation function for all nodes.</li>

</ol>

Denote by H the family of linear separators in R<em><sup>d</sup></em>, we have seen that the output function of any single node <em>i </em>in the <em>t</em>-th layer implements a function which is a member of H. Seen as a whole function, each layer implements a function from R<em><sup>d </sup></em>to R<em><sup>d</sup></em>:

<em>f</em>(<em>t</em>+1)(<em>z</em><em>t</em>) := <em>z</em><em>t</em>+1 = <em>h</em>(<em>W</em>(<em>t</em>+1)<em>z</em><em>t </em>+ <em>b</em>(<em>t</em>))

where <em>h </em>operates element-wise. Denote by F the class of such functions.

<ul>

 <li>Show that Π for every <em>n </em>≥ <em>d </em>+ 1.</li>

 <li>Express C in terms of H. Give a bound on the growth function of C, for <em>n </em>≥ <em>d </em>+ 1.</li>

 <li>Let <em>N </em>by the number of parameters in a multilayer NN as defined above. Express <em>N </em>in terms of <em>d </em>and <em>L </em>(number of layers).</li>

 <li><strong> </strong>Show that 2<em><sup>n </sup></em>≤ (<em>en</em>)<em><sup>N </sup></em>→ <em>n </em>≤ 2<em>N </em>log<sub>2</sub>(<em>eN</em>).</li>

 <li>We are finally in a position to derive a bound for the VC-dimension. Show that Π<sub>C</sub>(<em>n</em>) ≤ (<em>en</em>)<em><sup>N</sup></em>, and use this to show that <em>V Cdim</em>(C) ≤ 2<em>N </em>log<sub>2</sub>(<em>eN</em>).</li>

</ul>

<ol start="2">

 <li><strong> Suboptimality of ID3. </strong>Solve exercise 2 in chapter 18 in the book: Understanding Machine Learning: From Theory to Algorithms.</li>

 <li><strong> AdaBoost. </strong>Let <em>x</em><sub>1</sub><em>,…,x<sub>m </sub></em>∈ R<em><sup>d </sup></em>and <em>y</em><sub>1</sub><em>,…,y<sub>m </sub></em>∈ {−1<em>,</em>1} its labels. We run the AdaBoost algorithm as given in the recitation, and we are in iteration <em>t</em>. Assume that

  <ul>

   <li><strong>(Do not submit) </strong>Show that. Use the latter equalities to show that</li>

   <li>Show that the error of the current hypothesis relative to the new hypothesis is exactly1<em>/</em>2, that is:</li>

  </ul></li>

</ol>

<em>.</em>

<ul>

 <li>Show that AdaBoost will not pick the same hypothesis twice consecutively; that is<em>h<sub>t</sub></em>+1 6= <em>h<sub>t</sub></em>.</li>

 <li>Show that setting the weights to be brings <em>Z<sub>t </sub></em>to a minimum.</li>

</ul>

<ol start="4">

 <li><strong> Sufficient Condition for Weak Learnability. </strong>Let <em>S </em>= {(<em>x</em><sub>1</sub><em>,y</em><sub>1</sub>)<em>,…,</em>(<em>x<sub>n</sub>,y<sub>n</sub></em>)} be a training set and let H be a hypothesis class. Assume that there exists <em>γ &gt; </em>0, hypotheses <em>h</em><sub>1</sub><em>,…,h<sub>k </sub></em>∈ H and coefficients = 1 for which the following holds:</li>

</ol>

<em>k</em>

<em>y<sub>i </sub></em><sup>X</sup><em>a<sub>j</sub>h<sub>j</sub></em>(<em>x<sub>i</sub></em>) ≥ <em>γ                                                                              </em>(1)

<em>j</em>=1

for all (<em>x<sub>i</sub>,y<sub>i</sub></em>) ∈ <em>S</em>.

<ul>

 <li>Show that for any distribution <em>D </em>over <em>S </em>there exists 1 ≤ <em>j </em>≤ <em>k </em>such that</li>

</ul>

<em>.</em>

(Hint: Take expectation of both sides of inequality (1) with respect to <em>D</em>.)

<ul>

 <li>Let <em>S </em>= {(<em>x</em><sub>1</sub><em>,y</em><sub>1</sub>)<em>,…,</em>(<em>x<sub>n</sub>,y<sub>n</sub></em>)} ⊆ R<em><sup>d </sup></em>× {−1<em>,</em>1} be a training set that is realized by a <em>d</em>-dimensional hyper-rectangle classifier, i.e., there exists a <em>d </em>dimensional hyper-rectangle [<em>a</em><sub>1</sub><em>,b</em><sub>1</sub>] × ··· × [<em>a<sub>d</sub>,b<sub>d</sub></em>] that classifies the data correctly. Let H be the class of decision stumps of the form</li>

</ul>

<em> ,</em>

for 1 ≤ <em>j </em>≤ <em>d </em>and <em>θ </em>∈ R ∪ {∞<em>,</em>−∞} (for <em>θ </em>∈ {∞<em>,</em>−∞} we get constant hypotheses which predict always 1 or always −1). Show that there exist <em>γ &gt; </em>0, <em>k &gt; </em>0, hypotheses <em>h</em><sub>1</sub><em>,…,h<sub>k </sub></em>∈ H and <em>a</em><sub>1</sub><em>,…,a<sub>k </sub></em>≥ 0 with = 1, such that the condition in inequality

(1) holds for the training set <em>S </em>and hypothesis class H.

(Hint: Set <em>k </em>= 4<em>d </em>− 1 and let 2<em>d </em>− 1 of the hypotheses be constant.)

<ol start="5">

 <li><strong>(15 points, 7.5 points for each section) Linear regression with dependent variables. </strong>Consider the regression problem where <em>X </em>is a <em>n</em>×<em>d </em>data matrix, <em>y </em>is a column vector of size <em>n</em>, and <em>w </em>is a column vector of size <em>d </em>of coefficients. As we discussed in the lecture, if there are dependent variables there are infinite possible solutions that achieve this minimum. One sensible criterion to choose one among all possible solutions, is to prefer a solution with a minimal <em>`</em><sub>2 </sub> That is, we search for <em>w </em>that solves the following problem:</li>

</ol>

argmink<em>w</em>k<sup>2 </sup><em>w</em>

<em>s.t.Xw </em>= <em>y</em>

Assume that <em>d &gt; n </em>and that the matrix <em>X </em>has rank <em>n </em>(note that it in principle the rank can be smaller than <em>n</em>). And denote by <em>w<sup>? </sup></em>the optimum of the above problem.

<ul>

 <li>(No need to submit) Convince yourself that there exists a <em>w </em>such that <em>Xw </em>= <em>y</em>. Namely, the above problem has at least one feasible solution.</li>

 <li>Show that the optimal <em>w </em>can be written as a linear combination of the data point. Namely, there exists a vector <em>α </em>∈ R<em><sup>n </sup></em>such that the solution is given by <em>w<sup>? </sup></em>= <em>X<sup>T </sup>α</em>.</li>

 <li>Show that you can calculate <em>x<sup>T </sup>w<sup>? </sup></em>for all <em>x </em>by using only dot products between <em>x </em>∈ R<em><sup>d </sup></em>(hint: express the solution using the kernel matrix <em>K<sub>S </sub></em>= <em>XX<sup>T </sup></em>).</li>

</ul>

Note that the above implies that you can use the “kernel trick” in this case. Namely, you can also work with features <em>φ</em>(<em>x</em>) as long as you can calculate the corresponding kernel.

<em>Handout Homework 6: May 24, 2020                                                                                                                                  </em>3

<h1>Programming Assignment</h1>

Submission guidelines:

<ul>

 <li>Download the supplied files from Moodle (2 python files and 1 gz file). Details on every file will be given in the exercises. You need to update the code only in the skeleton files, i.e., the files that have a prefix ”skeleton”. Written solutions, plots and any other non-code parts should be included in the written solution submission.</li>

 <li>Your code should be written in Python 3.</li>

 <li>Make sure to comment out or remove any code which halts code execution, such as matplotlib popup windows.</li>

 <li>Your code submission should include these files: py,process data.py</li>

</ul>

<ol>

 <li><strong>(30 points) AdaBoost. </strong>In this exercise, we will implement AdaBoost and see how boosting can be applied to real-world problems. We will focus on binary sentiment analysis, the task of classifying the polarity of a given text into two classes – positive or negative. We will use movie reviews from IMDB as our data.</li>

</ol>

Download the provided files from Moodle and put them in the same directory:

<ul>

 <li>review polarity.tar.gz – a sentiment analysis dataset of movie reviews from IMBD.<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a>Extract its content in the same directory (with any of zip, 7z, winrar, etc.), so you will have a folder called review polarity.</li>

 <li>process data.py – code for loading and preprocessing the data.</li>

 <li>py – this is the file you will work on, change its name to adaboost.py before submitting.</li>

</ul>

The main function in adaboost.py calls the parse data method, that processes the data and represents every review as a 5000 vector <em>x</em>. The values of <em>x </em>are counts of the most common words in the dataset (excluding stopwords like “a” and “and”), in the review that <em>x </em>represents. Concretely, let <em>w</em><sub>1</sub><em>,…,w</em><sub>5000 </sub>be the most common words in the data, given a review <em>r<sub>i </sub></em>we represent it as a vector <em>x<sub>i </sub></em>∈ <em>N</em><sup>5000 </sup>where <em>x<sub>i,j </sub></em>is the number of times the word <em>w<sub>j </sub></em>appears in <em>r<sub>i</sub></em>. The method parse data returns a training data, test data and a vocabulary. The vocabulary is a dictionary that maps each index in the data to the word it represents (i.e. it maps <em>j </em>→ <em>w<sub>j</sub></em>).

(a) <strong>(10 points) </strong>Implement the AdaBoost algorithm in the run adaboost function. The class of weak learners we will use is the class of hypothesis of the form:

<em> ,</em>

That is, comparing a single word count to a threshold. At each iteration, AdaBoost will select the best weak learner. Note that the labels are {−1<em>,</em>1}. Run AdaBoost for <em>T </em>= 80 iterations. Show plots for the training error and the test error of the classifier implied at each iteration <em>t</em>, <em>sign</em>(<sup>P<em>t</em></sup><em><sub>j</sub></em><sub>=1 </sub><em>α<sub>j</sub>h<sub>j</sub></em>(<em>x</em>)).

<h2>4                                                                                                                                  Handout Homework 6: May 24, 2020</h2>

<ul>

 <li><strong>(10 points) </strong>Run AdaBoost for <em>T </em>= 10 iterations. Which weak classifiers the algorithm chose? Pick 3 that you would expect to help to classify reviews and 3 that you did not expect to help, and explain possible reasons for the algorithm to choose them.</li>

 <li><strong>(10 points) </strong>In next recitation you will see that AdaBoost minimizes the average exponential loss:</li>

</ul>

<em>.</em>

Run AdaBoost for <em>T </em>= 80 iterations. Show plots of <em>` </em>as a function of <em>T</em>, for the training and the test sets. Explain the behavior of the

<a href="#_ftnref1" name="_ftn1">[1]</a> http://www.cs.cornell.edu/people/pabo/movie-review-data/