# Predict Deep Proteome
We have: 
- samples of different KnockOuts (KO) where proteins are measured
- networks (PPIs, GRNs, ...)

We want to use high abundand proteins to predict low abundant proteins. 

We try this with different approaches:
- different ML models, parameters, ...
- only highest abundant proteins as features and predict low abundant
- use first neighbors as features to predict the protein

## main function is MultiLin5k
helper functions are ML_LinearRegression and help_5kData_analysis



### Data cannot be published yet though

