# Linear_Regression_Model

Research Question

Which quantitative variables are most strongly correlated with customer churn and can help to build a multiple regression model for prediction? Which variables then, will provide the most control over customer churn?

Objectives and Goals

The objectives of this analysis are threefold. First, to discover those variables which have the highest correlation with customer churn. Second, to build a predictive model that can provide insight into potential treatments for churn. Third, to provide a sound case for any certain action that may lead to the realization of the organization’s goal of lowering customer churn.

Summary of Assumptions

Certain assumptions intrinsic to the multiple regression approach to predictive modelling and analysis are: that the correlation between coefficients is linear (multivariate normality), variance of error are the same between dependent and independent variables (homoscedasticity), that the correlation is linear, and that there are no redundancies within the dataset amongst independent variables (multicollinearity). Let it be noted, if there are redundancies, this will increase the r squared value leading to a less precise prediction with the model.

Goals

Determine the correlation of each variable as it pertains to the target variable churn.

The following code was used to determine the correlation of each variable in respect to churn:
df_corr = df_copy[df_copy.columns[0:100]].corr()['Churn'][:-1].sort_values()
df_corr

Build the multiple regression model from the delimited columns. This step will align with the research question by allowing for insight into which variables are most correlated with customer churn and how the company can possibly have direct effect on customer churn.


Stats models OLS tool will be used to construct the model within the python environment. The code used is:
X=ddff[[  . . . ]].values
y = ddff[['Churn']].values
i = sm.add_constant(X)
l = sm.OLS(y,i)
fitted = l.fit()

Summary Statistics

There are many different aspects that make up a summary statistic. Those being: dependent variable, r-squared, adj. r-squared, model, method, f-statistic, probability f-statistic, log-likelihood, number of observations, AIC, BIC, data frame residuals, coefficient, standard error, p-value, and constant. 

R-Squared: 

This statistic refers to the sum of difference between the actual and expected values pertaining to the target variable. For the dataset, without modification, the r-squared value is .399. The range for this variable lies between zero and positive one. The closer the statistic is to one, the more accurate the model is of predicting the target value.

Adjusted R-Squared:

This statistic accounts for predictors not significant to the target value. The adjusted r-squared value for this data set is .398. Like the r-squared value, the range lies between zero and one.

Coefficient:

This number is a ratio of the dispersion of spread between probabilities of a points vector location within the model. The coefficient more accurately put is a representation of the mean value compared to the standard deviation. 

Target Variable

The target variable is the variable within the data set that needs to be predicted. In other words, all other variables will point to its value. The research question clearly states the desirable target variable is customer churn.

Predictor Variables

Predictor variables are the variables used to find the target variable. Simply put, every other variable relevant to the correlation/association of the target variable. To find the predictor variables, a correlation matrix was made as described above in the data goals section. The predictor variables are tenure, bandwidth GB per year, streaming movies, streaming tv, city, multiple, techie, device protection, online backup, job, payment method, tech support, yearly equipment failure, item 5, state, item 2, marital, item 3, case order, interaction, UID, contract, monthly charge, techie, payment method, email, online security, area, internet service, gender, and phone. Due to their high correlation and r-squared values in respect to churn, they will have been selected for the regression model.

Steps to Prepare the Data

1.	Import Packages

2.	Explore Physical Data Frame

3.	Create a Copy of the Data Frame

4.	Transform all columns to a useable form for the regression model and correlation matrix.

5.	Create a Multiple Linear Regression Model with all variables (associated and unassociated)

6.	Display Summary Statistics

7.	Create Correlation Matrix and Display Heatmap Visualization

8.	Create a Correlation Matrix of each Variable as it Pertains to Churn

9.	Visualize the Matrix

10.	Find Correlation within Data in Respect to Churn

11.	Drop Variables with Low Associative Properties

12.	Create Final Correlation Model

13.	Visualize the Final Correlation Matrix

14.	Check Eigen Values for Multicollinearity

15.	Evaluate the Model with the R-Squared Test

16.	Programmatically Create and Download a Copy of the Fully Prepared Data Set


### Initial Model

The initial model with code, summary and graphic are provided below.

Code:
a = df_copy[['CaseOrder', 'Zip','Lat','Lng','Population','Children','Age','Income','Outage_sec_perweek','Email','Contacts','Yearly_equip_failure','Tenure','MonthlyCharge','Bandwidth_GB_Year','item1','item2','item3','item4','item5','item6','item7','item8']].values
b = df_copy[['Churn']].values
c = sm.add_constant(a)
d = sm.OLS(b,c)
fitted = d.fit()

Summary Statistics:

Dep. Variable:	y	R-squared:	0.479
Model:	OLS	Adj. R-squared:	0.477
Method:	Least Squares	F-statistic:	195.0
Date:	Thu, 19 Nov 2020	Prob (F-statistic):	0.00
Time:	18:54:59	Log-Likelihood:	-2745.3
No. Observations:	10000	AIC:	5587.
Df Residuals:	9952	BIC:	5933.
Df Model:	47		
Covariance Type:	nonrobust		
	coef	std err	t	P>|t|	[0.025	0.975]
const	-0.5181	0.111	-4.647	0.000	-0.737	-0.300
x1	-0.3454	0.074	-4.647	0.000	-0.491	-0.200
x2	-8.82e-09	1.1e-08	-0.799	0.424	-3.05e-08	1.28e-08
x3	0.1727	0.037	4.647	0.000	0.100	0.246
x4	0.1727	0.037	4.647	0.000	0.100	0.246
x5	-1.988e-07	2.34e-06	-0.085	0.932	-4.78e-06	4.38e-06
x6	-0.0002	0.000	-0.707	0.479	-0.001	0.000
x7	4.428e-06	8.2e-06	0.540	0.589	-1.16e-05	2.05e-05
x8	1.774e-07	3.03e-07	0.585	0.559	-4.17e-07	7.72e-07
x9	2.798e-05	0.001	0.043	0.965	-0.001	0.001
x10	0.0003	0.001	0.515	0.606	-0.001	0.001
x11	-1.216e-07	2.36e-07	-0.514	0.607	-5.85e-07	3.42e-07
x12	0.0020	0.004	0.520	0.603	-0.006	0.010
x13	0.0009	0.002	0.466	0.641	-0.003	0.005
x14	1.386e-05	1.77e-05	0.784	0.433	-2.08e-05	4.85e-05
x15	-0.0072	0.002	-4.565	0.000	-0.010	-0.004
x16	0.0010	0.000	6.157	0.000	0.001	0.001
x17	6.855e-08	1.13e-07	0.604	0.546	-1.54e-07	2.91e-07
x18	-0.0011	0.002	-0.476	0.634	-0.006	0.003
x19	-0.0013	0.006	-0.213	0.832	-0.013	0.010
x20	-0.0005	0.001	-0.465	0.642	-0.003	0.002
x21	0.0007	0.001	0.617	0.537	-0.001	0.003
x22	0.0042	0.003	1.308	0.191	-0.002	0.011
x23	-0.0065	0.005	-1.288	0.198	-0.016	0.003
x24	-0.0695	0.009	-8.124	0.000	-0.086	-0.053
x25	0.1300	0.004	33.962	0.000	0.122	0.137
x26	-0.0111	0.006	-1.731	0.083	-0.024	0.001
x27	0.0072	0.007	1.036	0.300	-0.006	0.021
x28	0.0612	0.007	8.369	0.000	0.047	0.076
x29	0.0245	0.011	2.227	0.026	0.003	0.046
x30	0.0308	0.013	2.294	0.022	0.004	0.057
x31	0.0408	0.007	5.934	0.000	0.027	0.054
x32	0.0576	0.010	5.489	0.000	0.037	0.078
x33	0.0400	0.008	4.986	0.000	0.024	0.056
x34	0.0309	0.008	3.864	0.000	0.015	0.047
x35	0.0120	0.017	0.708	0.479	-0.021	0.045
x36	0.0028	0.020	0.140	0.888	-0.037	0.043
x37	-0.0129	0.007	-1.990	0.047	-0.026	-0.000
x38	-0.0096	0.003	-3.424	0.001	-0.015	-0.004
x39	-0.0284	0.001	-20.840	0.000	-0.031	-0.026
x40	0.0039	0.000	10.852	0.000	0.003	0.005
x41	0.0003	1.64e-05	15.488	0.000	0.000	0.000
x42	-0.0018	0.005	-0.395	0.693	-0.011	0.007
x43	-0.0046	0.004	-1.064	0.287	-0.013	0.004
x44	0.0018	0.004	0.457	0.648	-0.006	0.010
x45	-0.0037	0.004	-1.041	0.298	-0.011	0.003
x46	-0.0005	0.004	-0.129	0.898	-0.008	0.007
x47	0.0015	0.004	0.388	0.698	-0.006	0.009
x48	0.0006	0.004	0.166	0.868	-0.006	0.008
x49	-0.0016	0.003	-0.466	0.641	-0.008	0.005
Omnibus:	931.681	Durbin-Watson:	1.971
Prob(Omnibus):	0.000	Jarque-Bera (JB):	338.181
Skew:	0.203	Prob(JB):	3.67e-74
Kurtosis:	2.196	Cond. No.	9.14e+16


Justification of Model Reduction

The approach and justification for predictor variable dissociation with the target variable used many different statistical tests. First, correlation via a correlation matrix was used in order to determine the relationship between all predictor variables and the target. Second, the r-squared value returned by the R-Squared statistical test was used to determine direct correlation to the target variable. This test also revealed the spacing of the coefficients of X in terms of y (churn) to help illuminate any multicollinearity. Third, visualizations were used to explore the relationship between different predictors and the target. These visuals were univariate, bivariate, and correlation visualizations. Fourth, Eigen values were determined for each predictor value resulting in the elimination of those variables suggesting multicollinearity. Fifth, and last, the final model’s r-squared value was compared to that of the original model and the change was insignificant. This means that the elimination of those columns deemed disassociated did not harm the model.

### Reduced Multiple Regression Model

The final model with code, summary and graphic are provided below.

Code:
X = ddff[['Tenure','Bandwidth_GB_Year','StreamingMovies','StreamingTV','City','Multiple','Techie','DeviceProtection','OnlineBackup','Job','PaymentMethod','TechSupport','Yearly_equip_failure','item5','State','item2','Marital','item3','CaseOrder','Interaction','UID','Contract','MonthlyCharge','Techie','PaymentMethod','Email','OnlineSecurity','Area','InternetService','Gender','Phone']].values
y = ddff[['Churn']].values
i = sm.add_constant(X)
l = sm.OLS(y,i)
fitted = l.fit()

Summary Statistics:

array([-7.30969841e+01, -5.66786150e+00,  5.02224604e-02, -4.87315598e+01,
        2.43654243e+01,  2.43654243e+01,  6.00734656e-01])
Out[300]:
OLS Regression Results
Dep. Variable:	y	R-squared:	0.394
Model:	OLS	Adj. R-squared:	0.393
Method:	Least Squares	F-statistic:	1622.
Date:	Sat, 21 Nov 2020	Prob (F-statistic):	0.00
Time:	11:50:13	Log-Likelihood:	-56491.
No. Observations:	10000	AIC:	1.130e+05
Df Residuals:	9995	BIC:	1.130e+05
Df Model:	4		
Covariance Type:	nonrobust		
	coef	std err	t	P>|t|	[0.025	0.975]
const	-73.0970	1.852	-39.477	0.000	-76.727	-69.467
x1	-5.6679	0.232	-24.408	0.000	-6.123	-5.213
x2	0.0502	0.003	18.115	0.000	0.045	0.056
x3	-48.7316	1.234	-39.479	0.000	-51.151	-46.312
x4	24.3654	0.617	39.475	0.000	23.156	25.575
x5	24.3654	0.617	39.475	0.000	23.156	25.575
x6	0.6007	0.018	32.715	0.000	0.565	0.637
Omnibus:	660.396	Durbin-Watson:	1.965
Prob(Omnibus):	0.000	Jarque-Bera (JB):	352.118
Skew:	0.300	Prob(JB):	3.46e-77
Kurtosis:	2.304	Cond. No.	1.02e+23


Model Comparison

Logic of the Variable Selection Technique

The r-squared is a common model evaluation metric for predictor selection. It allows for the insight into which predictor variables are of correlated space within terms of the target. The k-mean and k-value are also great representations of this space. However, the r-squared technique was used because of its simplicity and ability to easily point out variables disassociated to the target. 

Eigen values are another great way to determine desirability within predictors. This is because they flag predictors that may have multicollinearity to the target. This makes the selection of relevant data simple and easy for any linear regression model.

Using these model evaluation metrics alongside correlation metrics, the predictor variables were reduced to the final model’s container.

Model Evaluation Metric

1.	R-Squared Test

regression = LM.LinearRegression(normalize=False, fit_intercept=True)

def r2_est(a,b):
    return r2_score(b,regression.fit(a,b).predict(a))

print ('Baseline R2: %0.3f' %  r2_est(a,b))
r2_impact = list()
for l in range(a.shape[1]):
    sel = [z for z in range(a.shape[1]) if z!=l]
    r2_impact.append(((r2_est(a,b) -r2_est(a[:,sel],b)) ,df_copy.columns[l]))
    
for imp, varname in sorted(r2_impact, reverse=True):
    print ('%6.3f %s' %  (imp, varname)
    
2.	Eigen Values

ll = ddff[['Churn','Tenure','Bandwidth_GB_Year','StreamingMovies','StreamingTV','City','Multiple','Techie','DeviceProtection','OnlineBackup','Job','PaymentMethod','TechSupport','Yearly_equip_failure','item5','State','item2','Marital','item3','CaseOrder','Interaction','UID','Contract','MonthlyCharge','Techie','PaymentMethod','Email','OnlineSecurity','Area','InternetService','Gender','Phone']]
correlation = np.corrcoef(ddff, rowvar=0)
eigenvalues, eigenvectors = np.linalg.eig(correlation)
display(eigenvalues)

3.	Correlation Matrix

ddff = df_copy[['Churn','Tenure','Bandwidth_GB_Year','StreamingMovies','StreamingTV','City','Multiple','Techie','DeviceProtection','OnlineBackup','Job','PaymentMethod','TechSupport','Yearly_equip_failure','item5','State','item2','Marital','item3','CaseOrder','Interaction','UID','Contract','MonthlyCharge','Techie','PaymentMethod','Email','OnlineSecurity','Area','InternetService','Gender','Phone']]
df_corr = ddff[ddff.columns[0:70]].corr()['Churn'][:].sort_values()
df_corr


Results

Regression Equation:

y = -7.30969841e+01 (Intercept)  - 5.66786150e+00 (Tenure) +  5.02224604e-02 (Bandwidth GB per Year) - 4.87315598e+01 (Case Order) + 2.43654243e+01 (Interaction) + 2.43654243e+01 (UID) + 6.00734656e-01 (Monthly Charge)

Interpretation of Coefficients:

The coefficients represent a predictors value as the partnering variables equal zero. For example, when bandwidth, case order, interaction, UID, and monthly charge are equal to zero; then tenure is equal to - 5.66786150e+00 plus the intercept. Using the above equation allows for the plugging in of predictor variables to assume the value y of the target.

Statistical and Practical Significance of the Model:

The correlation matrix, r-squared, and adjusted r-squared values provide insight into the statistical and practical significance of the model. First, the correlation matrix shows a low correlation for each predictor variable as it pertains to the variable in question churn. The highest correlation throughout the dataset to churn is tenure at -.485475. This shows that from the beginning, the correlation necessary to answer the business’ question of lowering customer churn just was not present in the data provided. Second, the r-squared and adjusted r-squared values sit around .394 and .393. This is terribly low. In a perfect dataset, the predictors r-squared values would equal a perfect 1. The lowest any reliable r-squared value will dip is low seventies (.70). The closer these two are to zero, the less accurate the model is to the data. 
Practically, the model is insignificant. It provides, in simple terms, a 39.4% confidence of using the predictor variables to assume the target variable. 

Recommendations

From the analysis and model, some recommendations are inherent to the data. There is an absolute relationship represented between tenure, bandwidth, monthly charge, and the target variable churn. The understood associations within the data are as follows:

1.	Churn: Tenure

a.	Negative Association

The higher the tenure, the lower the churn associated is. This means that those with early tenure are needing strategic management and support to ensure longevity.

2.	Churn: Monthly Charge

a.	Negative Association

The higher the monthly charge, the higher the associated churn. This means in order to keep customers from churning, success will be gathered in a lower premium.

3.	Churn: Bandwidth

a.	Positive Association

The higher the bandwidth the lower the churn. This makes sense as the more bandwidth there is, the better the internet service will be. Thus, intuitively increasing satisfaction with the service. The takeaway is to increase bandwidth for customer in order to decrease churn.

The final suggestion that the analysis points so obviously to is a re-acquiring of data. There are obvious red flags with each variable in the data. There are issues which cannot be imputed, they need to be resampled for the integrity of the data. Some examples are listed.

1.	Area categories urban, suburban and rural have matching descriptive statistics for population. Rural areas do not have the same populations as urban hence their categorical classification. 

2.	There are a great number of negative entries for bandwidth and outage seconds per week. There is no such metaphysical possibility for negative bandwidth or outage seconds per week in the service industry. The assumption is that these two specific variables were gathered internally, therefore they should be accurate as they are contractual obligations and service records. These variables must be re-acquired. 


