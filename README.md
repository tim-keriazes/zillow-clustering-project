![Alt text](C:\Users\Timmy\DownloadsZillow Offers exterior sign-1000)
# Logerror Regression Project With Clustering Methods: Zillow Data Set

___
___

# <a name="scenario"></a>Scenario
Utilizing the 2017 Zillow data set, I set out to build a model that would predict the logerror of single family homes, examining the correlation between the data points and our target variable (logerror). I will identify critical features that can be used in a regression model to predict logerror as well as incorporating feature engineering utilizing clustering methods. 
___

# <a name="project-planning"></a>Project Planning
### Goal:
The goal for this project is to create a model that will accurately predict a home's logerror determined by the county's Appraisal District. To do so, I will have to identify which of the various features are drivers to this end, and implement them into a machine learning regression model. 

### Initial Hypotheses:

>$Hypothesis_{1}$
>
> There is a relationship between county location and logerror.


>$Hypothesis_{2}$
>
> There is a correlation between number of luxury home items such as pool, garage, number of rooms and square footage.

### Project Planning Initial Thoughts:
- My initial thoughts were that there would be a direct relationship to the location of the property and its logerror.
- Additionally, I believe there to be a correlation, not just to the number of bedrooms and bathrooms, but also to a ratio of square footage per number of rooms and what I called luxury items such as a pool, garage, air conditioning etc.
- To explore this I created a new feature:
    - df['luxury_score'] = (df['poolcnt']+df['garagecarcnt']+df['bedrooms']+df['bathrooms']+df['airconditioning_encoded'])
___
# <a name="key-findings"></a>Key Findings

## Exploration Takeaways
After removing data set outliers, especially those homes with more than 6 bedrooms, 7 bathrooms, over a value of $3,000,000, and with a square footage of more than 10,000 sqft. The majority of homes are in the bottom 20% of the overall square footage and bottom 45% of home_value. 

Mean logerror = 0.01816
RMSE baseline = .175
Best model = Tweedie Regressor with RMSE .175

___
# <a name="tested-hypotheses"></a>Tested Hypotheses

## Key Takeaways/Summary of stats tests

Hypothesis Testing
Tested Hypotheses

### 1. Ho: Log error will be the same across all quarters observed.
    Ha: Log error will vary based quarter observed.
    -We reject the null hypothesis.

### 2. Ho: Log error will be the same across all three counties.
    Ha: Log error will vary based on county.
    -We reject the null hypothesis.

### 3. Ho: There is no correlation between tax rate and log error.
    Ha: There is a relationship between tax rate and log error.
    -We fail to reject the null hypothesis. There is a relationship between tax rate and log error

### 4. Ho: There is no correlation between longitude and log error.
    Ha: There is a relationship between longitude and log error.
    -We reject the null hypothesis. Strong relationship between longitude and logerror

### 5. Ho: There is no correlation between cost per sqft and log error.
    Ha: There is a relationship between cost per sqft and log error.
    -We reject the null hypothesis. Strong relationship between cost_per_sqft and logerror

### 6. Ho: There is no correlation between luxury_sqft_per_age and log error.
    Ha: There is a relationship between luxury_sqft_per_age and log error.
   -We reject the null hypothesis. Strong relationship between luxury_sqft_per_age and log error.

In Summary:
Following the evaluation of the different models using my selected features, I found that the best performing model was the Tweedie Regressor
It performs best when at the median logerror.

___
# <a name="data-dictionary"></a>Data Dictionary
|                   column_name                   |                                                       description                                                       |                   key                  |       dtype      |
|:-----------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------:|:--------------------------------------:|:----------------:|
| `bathrooms` / `bathroomcnt`                     |    Number of bathrooms in home including fractional bathrooms                                                           |                                        | float64          |
| `bedrooms` /  `bedroomcnt`                      |    Number of bedrooms in home                                                                                           |                                        | float64          |
| `sqft` / `calculatedfinishedsquarefeet`  |    Calculated total finished living area of the home                                                                    |                                        | float64          |
|   `fips`        |    Federal Information Processing Standard code  |                                        | float64 /  int64 |
| `cost_per_sqft`                                      | Unique identifier for parcels (lots)                                                                                    |                                        | int64            |
|   `yearbuilt`                                   |    The Year the principal residence was built                                                                           |                                        | float64 /  int64 |
| `home_value` / `taxvaluedollarcnt`              |   The total tax assessed value of the parcel                                                                            |                                        | float64          |
| `taxamount`                           | The total property tax assessed for that assessment year                                                                |                                        | float64          |
| `sqft_room_ratio`                                        | The ratio of sqft to total number of bedrooms and bathrooms.                                                                                     |                                        | int64       |
|'luxury_score'                | count of pool, car garagem bedrooms, bathrooms, aircondition | | int64|
|'luxury_sqft_per_age'| luxury score times sqft divided by age| | int64|
|'age'                | 2017 - 'yearbuilt' | | int64|
|'home_value_structure_tax_difference'                | 'home_value'-'structuretaxvaluedollarcnt' | | int64|
|'taxrate'                | 'taxamount'/'home_value' | | int64|
|'cost_per_sqft'               | 'homevalue'/'sqft' | | int64|
|'sqft_room_ratio'                | 'sqft'/'number of beds and baths' | | int64|
|'taxrate'                | 'taxamount'/'home_value' | | int64|
|'taxrate'                | 'taxamount'/'home_value' | | int64|



___
# <a name="workflow"></a>Workflow

1. prep-your-repo (env file and gir ignore to utilize mysql credentials)
1. import all necessary libraries and additional python files (wrangle.py which is available for reference in my repo)
1. acquire-data (wrangle.py)
1. clean-prep-and-split-data (wrangle.py)
1. explore-data
    - hypothesis-testing
1. evaluate-data
1. Develop clustering models
    -add as features
3. modeling
    - identify-baseline
    - train-validate
    - test
    
