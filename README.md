---
title: Fit DIVA to SHJ Project
---


This project is meant to fit the DIVergent Autoencoder model of human categorization (DIVA; Kurtz 2007) to the human behavioral data collected by Nosofsky (1994), replicating the classic Shepard, Hovland and Jenkins experiment that looked at learning difficulty of six different types of rule-based category structures.

Item Prediction Vs Response probability methods for representing human performance-->
There are two ways we could represent human performance on each training block.  One would be to have the model generate a definite prediction for each item and record that prediction as correct or incorrect (each item has an acuracy of 1 or 0). Once all item predictions have been recorded for the epoch, we can average the binary item-wise accuracies to get the epoch-wise accuracy.  The other way is to record response probabilities instead of definite predictions.  For each item the model will generate probabilities in regard to which category the item belongs to.  If for a certain item, the model predicts a 0.80 chance of that item belonging to the correct category, this could be interpretated as predicting on average humans will get this item correct 80% of the time at this stage of learning.  Once all the response probabilites are recorded for the epoch, they can be averaged and will constitute the epoch-wise accuracy.
-->included in the repository are versions for each of these two methods.

Orthogonal array tuning--> a way of using combinatorics to optimize the model's hyperparameters.

Two Different Response Functions-->
The tradition response function used for DIVA is (1/SSD[A])/((1/SSD[A])+(1/SSD[B])), which basically says that the channel/category that has the least error is most likely correct-- a simple version of the luce choice rule.

Dr. Nosofsy suggested to use P(A|i)  =    exp(-c*SSD[A]) /  {exp(-c*SSD[A] + exp(-c*SSD[B]}.  This is basically a softmax function, where c is a some scalar.

In the case where (1/SSD[A])/((1/SSD[A])+(1/SSD[B])) = exp(-c*SSD[A]) /  {exp(-c*SSD[A] + exp(-c*SSD[B]}, c=(ln(SSD[A]/SSD[B]))/( SSD[A]  - SSD[B]  )... if I'm doing my algebra right. So, the value of c that would make the two functions equivalent would be different for different values of SSD[A] and SSD[B].

We've been talking about two different methods for determining block accuracy.  One where each item ends with a definite prediction (1 or 0; by taking argmax of the response probabilities), and another where each item's accuracy is equal to the response probability generated for that item. 

I think this fancier response rule would impact the response probability method, while having basically no effect on the item prediction method.

For the fancy rule- when c is equal to 0 response probability will always be 50/50; when c is 1, the response probability will directly reflect the SSDs; when c is greater than 1, it will upregulate the probability of the 'winning' (channel with lower SSD) channel and downregulate the probability of the losing channel.

So if the probabilities without c are [.6,.4], then:  c=0-->[.5,.5]; c=1-->[.6,.4]; c>1--> something like [.8,.2].  If .8 is a better fit to the human accuracy on that item at that block, then the large c will make an improvement in the fit.  This is with the response probability method.

For the item prediction method, the only time c will change the outcome of the argmax is c<=0, where when 0 argmax will be indeterminant and less than 0 it will be inverse.


divaWrap_RespProbMethod_nosofskySuggestion.py uses this more complex response function, which adds an extra hyperparamteter (c).
