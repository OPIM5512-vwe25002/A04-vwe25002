# A04-vwe25002
A04_xAI

Name: Kyra Lim

Purpose: This code (model.py) runs two different models to predict loan approvals, evaluates feature importance, and creates PDP and ICE curves for the top 5 features of each model.

Data set: Customer Loan Data (train_loan_imbalanced.csv)

How to run: The script is written in Python and saved in the src folder. Open model.py and run the script using the play button in Visual Studio. You can also run using the following prompts in the terminal:
1. cd src
2. python model.py 

Outputs: feature importance plots for the Random Forest Classifier and ADABoost models, feature importance plots for the top 5 features for each model, and PDP and ICE curves for each of top 5 features for each model. All plots are saved as png files in the figures folder.

Write-Up:
I tried two different models for this assignment:
1. Random Forest Classifier
2. TPOT to generate the optimal pipeline yielding steps of:
	a. Normalizer (scales each row of the data so its values sum to 1)
	b. ADABoost Classifier
	(the other pipeline steps don't make changes to the data)

The accuracy of the two models was similar at 84% for the Random Forest Classifier and 81% for the ADABoost. The F1 score for the minority class (Loan Denial) for both models was low at 0.64 for Random Forest Classifier and 0.55 for the normalized ADABoost Classifier. In both cases, the F1 score was brought down by the low recall, meaning that the model wasn't good at catching actual denials. I suspect that this is driven by the significant class imbalance on a small data set. Only 29% or 28 loans in the data set were denied, so the model struggled to find signal for denials that were different from approvals.

I ran permutation importance for both models to find the most importance features for each. The two models had four of the top 5 most important features the same (Applicant Income, Coapplicant Income, Credit History, and Loan Amount). The Random Forest Classifier had Property Area Semiurban as the fifth most important feature, and the ADABoost Classifier had Gender_Male. 

Below are some observations I had on the plots related to the Random Forest Classifier:
	1. I was not surprised to see that the top five most important features were very similar but not the same between my models. If the features do hold a strong signal, they should emerge regardless of the model. However, some features will present differently because the models follow different processes and prioritize different things.

	2. I see that the most important feature by far is Credit History because it has the highest median on the feature importance plot. The x-axis on the feature importance plot represents the impact to the model's accuracy when values of the feature are shuffled (holding all other features constant). The y-axis is all of the model's features. The median value for feature importance for Credit History is ~0.14, which means that the model's accuracy dropped 14 percentage points for the median shuffle.

	3. The ICE curves for Property Area Semiurban form a large X. This is because both the target and feature variables are categorical. The ICE plots run the feature variable through a range of values between the min and max of the feature to see the model's predicted target value. The x-axis represents potential values of the feature (Semiurban Property), and the y-axis represents the predicted loan status. Since the only unique values are 0 and 1, the end points are the only thing that matters because the points on the diagonal aren't realistic possible values of the feature. The PDP is a slightly upwardly sloping diagonal line, meaning that semiurban properties are more likely to have approved loans (approval = 1).  

	4. The PDP curve for Applicant Income is interesting. It has a steep increase from incomes of $0 to $5,000, continues to rise from $5,000 to $20,000, and then levels off. Technically, there is a slight dip from $20,000 to $40,000. I suspect that the Applicant Income is either monthly or scaled down by a factor of 10. I would need a better understanding of the data to know for certain. Regardless, this tells me that as applicant incomes rise, especially for lower income folks, the likelihood of loan approval increases dramatically. 

	5. I was surprised to see that the PDP curve for Co-Applicant Income is downward sloping. That implies that as the Co-Applicant's income increases, the likelihood of loan approval decreases. One possible explanation of this phenomenon could be that only less creditworthy borrowers apply for loans with co-applicants. The less creditworthy applicant may have lower income themselves, so they rely on the co-applicant's income to bolster their odds of loan approval. It would be interesting to see if applicant income and co-applicant income are highly negatively correlated. 

	6. Comparing the PDP Curves for Loan Amount between the two models is also interesting. The x-axis is loan amount, the y-axis is the likelihood of loan approval, and the scale for both curves is the same. In both cases, the PDP curve is downward sloping, meaning that the likelihood of loan approval decreases as the loan amount increases. The PDP curve from the ADABoost is a smooth, reverse S-curve compared to the Random Forest Classifier, which is flatter and bumpier. It makes sense that the curves would look different because they make different assumptions but ultimately tell a similar story.