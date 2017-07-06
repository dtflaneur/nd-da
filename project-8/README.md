## P8 – Create a Tableau Story

Project url final : https://public.tableau.com/profile/pratj#!/vizhome/TItanicDataAnalysis/Story1
Project url v1: https://public.tableau.com/profile/pratj#!/vizhome/TItanicDataAnalysisv1/Story1

#### Summary:

The Titanic ship sank in the North Atlantic Ocean in 1912 after colliding with an iceberg, killing more than 1,500 of the 2,224 passengers aboard. 

In this excercise we will see the likely effect of several factors on the survival rate of passengers. We will start by gender. Then we will add passengers’ ages and classes to gender separately to see how these two combinations would affect survival rates.


#### Design:

Throughout the visualization, default color palletes are used. This appeared fine & looked consistent to make it easier to read the plots. 

The story consists of 5 pairs of slides. A dashboard is split into two sides, the left side shows a plot for total % of passengers by gender, and the right side show two plots based on the % of passengers survived or perished by Gender.

I mostly used bar charts. In almost all bar chat, the y-axis shows the count of passengers, and the labels on top of the bars show the percentage of females/males. 
Reasoning: I think the focus is basically on the count of people survived vs. perished, based on available data. Bar charts are best suited to show such a comparison between counts/frequency of occurrences of multiple categories. 


#### Feedback:

I showed the initial version of the visualization to a co-worker. Below is the received feedback and the changes made: 
1.	The Age distribution chart was quite cluttered, thus I converted it into a age group chart.

#### Resources:
The original dataset used is from the Data Set Options by Udacity selecting the Titanic Dataset.  The original dataset was cleaned to remove records with null age.

The cleaning process included:
-	Dropping features that were not adding value to the purpose of the needed analysis such as: 'Ticket', 'Fare', 'Cabin', 'Embarked'
-	Creating a new categorical feature: “Age Group”:



