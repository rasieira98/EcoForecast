![image](https://github.com/rasieira98/EcoForecast-23/assets/116558787/61327ed2-ab7b-458d-9f6e-fbd43d0597a8)
***
# INTRODUCCION
This GitHub repository contains a project aimed at calculating the energy surplus for each country in the next hour, addressing Schneider's challenge on the NUWE platform.

If you want to execute all pipelines with only one command:
```
./scripts/run_pipeline.sh 2022-01-01 2023-01-01 data data/raw_data.csv data/processed_data.csv models/model.pkl data/test_data.csv predictions/predictions.json
```
***
# DATA INGESTION
### Description
It collects data from the ENTSO-E Transparency API to get how the electricity consumption (Load) and the generation of different types of energy. 

In this script there are several steps and considerations which were kept in mind during the ingestion:

1. Download the data from 01-01-2022 to 01-01-2023
2. If it was necessary to download more data, there's a method to split the calls year by year (Max period to query using the API = 1 year).
3. In order to speed up the collection, we use threads (1 thread by country). The result is to collect all data in less than 5 minutes.
4. After the data extraction, we have a lot of files by country (Load and gen) splitted by every type of energy (Renewables and not green energy). Then, in order to gather all this data in only one file, a preprocessing was implemented. We build the raw_data.csv by using a pivot function. 
### Output
The output of this process has the following format:
```
Country,StartTime,EndTime,AreaID,UnitName,B01,B02,B03,B04,B05,B06,B07,B08,B09,B10,B11,B12,B13,B14,B15,B16,B17,B18,B19,B20,Load
DE,2022-01-01T00:00+00:00Z,2022-01-01T00:15+00:00Z,10Y1001A1001A83F,MAW,4325.0,3521.0,,2628.0,2057.0,279.0,,,26.0,568.0,1421.0,121.0,,3350.0,126.0,0.0,841.0,5795.0,24604.0,333.0,41804.0
```

### Logging
The script uses a custom logger named "Data_ingestion" to log its progress. Logs are informative and include details about data ingestion.
### Example Usage
```
python data_ingestion.py --start_time 2022-01-01 --end_time 2023-01-01
```

***
# DATA PREPROCESSING
***
### Description
In this script, we preprocess the data generated in the previous step of the pipeline. The steps which were considered for this process are described below:
1. Convert the 'StartTime' column to datetime format.
2. Sort the DataFrame based on the 'StartTime' column.
3. Filter out rows where 'AreaID' is null, because the API returned records with AreaID = null
4. We filtered the renewable columns, which are configured in the config.yaml file (Variable columns_clean)
5. Remove outliers from the input DataFrame based on Z-scores
6. Interpolate missing values by interval for each country. For example, if at 15:00 is null, 15:15 is null and 15:45 is null, but 15:30=100, then we interpolate this interval (Result is 15:00, 15:15 and 15:45 = 100). If it's empty, the result for this interval is None. During this process, we create a dummy dataframe with all intervals in two specifics dates (2022-01-01 and 2022-12-31 for this case) in order to have 8760 records by country (UK with a lot of nulls values).
7. Resample the DataFrame to hourly frequency. If all values are NaN, then the result is NaN, not 0.
8. Create columns 'green_energy' and 'surplus' based on specified calculations. If green_enery or Load is NaN, then the surplus is NaN too for that country.
9. Create a target table with columns 'StartTime' and 'target'.
10. Pivot the final table based on specified columns.
11. Merge the pivot table with the target table.
12. Finally, we decided not to filter the rows by using the dropna function, because we believe that in a real case, we would have to consider the bigger quantity of data. Therefore, 8760 records were considered for the next step.

### Output
The output of this process has the following format:
```
StartTime,B01_DE,B01_DK,B01_HU,B01_IT,B01_NE,B01_PO,B01_SE,B01_SP,B01_UK,B09_DE,B09_DK,B09_HU,B09_IT,B09_NE,B09_PO,B09_SE,B09_SP,B09_UK,B10_DE,B10_DK,B10_HU,B10_IT,B10_NE,B10_PO,B10_SE,B10_SP,B10_UK,B11_DE,B11_DK,B11_HU,B11_IT,B11_NE,B11_PO,B11_SE,B11_SP,B11_UK,B12_DE,B12_DK,B12_HU,B12_IT,B12_NE,B12_PO,B12_SE,B12_SP,B12_UK,B13_DE,B13_DK,B13_HU,B13_IT,B13_NE,B13_PO,B13_SE,B13_SP,B13_UK,B15_DE,B15_DK,B15_HU,B15_IT,B15_NE,B15_PO,B15_SE,B15_SP,B15_UK,B16_DE,B16_DK,B16_HU,B16_IT,B16_NE,B16_PO,B16_SE,B16_SP,B16_UK,B18_DE,B18_DK,B18_HU,B18_IT,B18_NE,B18_PO,B18_SE,B18_SP,B18_UK,B19_DE,B19_DK,B19_HU,B19_IT,B19_NE,B19_PO,B19_SE,B19_SP,B19_UK,Load_DE,Load_DK,Load_HU,Load_IT,Load_NE,Load_PO,Load_SE,Load_SP,Load_UK,green_energy_DE,green_energy_DK,green_energy_HU,green_energy_IT,green_energy_NE,green_energy_PO,green_energy_SE,green_energy_SP,green_energy_UK,surplus_DE,surplus_DK,surplus_HU,surplus_IT,surplus_NE,surplus_PO,surplus_SE,surplus_SP,surplus_UK,target
2022-01-01 00:00:00,17295.0,511.0,523.0,683.0,85.0,221.0,,524.0,,104.0,,0.0,640.0,,,,0.0,,2274.0,,,0.0,,0.0,,,,5690.0,,44.0,1951.0,0.0,199.0,,1046.0,,378.0,,36.0,200.0,,0.0,7086.0,746.0,,,,,,,,0.0,0.0,,504.0,,30.0,,,,,96.0,,0.0,1.0,0.0,0.0,0.0,0.0,0.0,75.0,,23112.0,1904.0,,,6884.0,,,0.0,,95800.0,1189.0,743.0,2140.0,5739.0,4071.0,4021.0,6456.0,,165125.0,3218.0,16457.0,19756.0,40706.0,13935.0,15331.0,19530.0,1244.0,145157.0,3605.0,1376.0,5614.0,12708.0,4491.0,11107.0,8943.0,,-19968.0,387.0,-15081.0,-14142.0,-27998.0,-9444.0,-4224.0,-10587.0,,DK
2022-01-01 01:00:00,17322.0,519.0,516.0,670.0,85.0,223.0,,532.0,,106.0,,0.0,639.0,,,,0.0,,1392.0,,,0.0,,0.0,,,,5683.0,,44.0,1831.0,0.0,216.0,,1045.0,,132.0,,36.0,155.0,,0.0,7088.0,733.0,,,,,,,,0.0,0.0,,504.0,,28.0,,,,,96.0,,0.0,1.0,0.0,0.0,0.0,0.0,0.0,75.0,,21798.0,1738.0,,,5518.0,,,0.0,,91464.0,1051.0,902.0,2233.0,5494.0,3997.0,3948.0,6144.0,,160415.0,3126.0,15426.0,18685.0,39465.0,13579.0,15270.0,18383.0,1131.0,138401.0,3309.0,1526.0,5528.0,11097.0,4436.0,11036.0,8625.0,,-22014.0,183.0,-13900.0,-13157.0,-28368.0,-9143.0,-4234.0,-9758.0,,DK
```

### Configuration
Ensure that the config.py file is properly configured with relevant settings for data processing.


### Logging
The script uses a custom logger named "Data_processing" to log its progress. Logs are informative and include details about data processing, such as different metrics in each stage.
### Example Usage
```
python data_processing.py --input_file data/raw_data.csv --output_file data/processed_data.csv
```

# TRAINING
### Overview
Is designed to streamline the process of train a model on a given dataset. The functionality is structured into modular functions that handle loading data, splitting data, train and save the model.
### Usage
1. Install Dependencies
Ensure that the required dependencies are installed. You can install them using the following command:
```
pip install pandas autogluon.tabular sklearn
```
2. Prepare Configuration
Make sure that the config.py file is properly configured. This file likely contains settings related to training process.

3. Run the Script
Execute the script from the command line with the following arguments:

```
python model_training.py --input_file path/to/processed_data.csv --model_file path/to/trained_model.pkl
```
- --input_file: Path to the CSV file containing the procesed data.
- --model_file: Path to save the pre-trained model file.
- 
The script will log its progress, and the final model will be saved in the specified output file.

### Script Structure
- load_data(file_path: str) -> pd.DataFrame:
Description: Load data from a CSV file and return it as a Pandas DataFrame.
- split_data(df: pd.DataFrame) -> pd.DataFrame:
Description: Split the input DataFrame into training and testing sets, save the sets as CSV files, and return the training set.
- train_and_save_model(df_train: pd.DataFrame, model_path: str):
Description: Train a model using the TabularPredictor from the `autogluon.tabular` module, and save the trained model to the specified path.
- parse_arguments():
Description: Parse command-line arguments for the model training script.
- main(input_file, model_file):
Description: Main function that orchestrates the entire train process. It loads data, train the model and save the model.

### Dependencies
- argparse: Used for parsing command-line arguments.
- pandas: Essential for working with tabular data.
- autogluon: Provides the TabularPredictor class for working with tabular data.
### Configuration
Ensure that the config.py file is properly configured with relevant settings for data processing.

### Logging
The script uses a custom logger named "Model_training" to log its progress. Logs are informative and include details about data loading, model training and model save.
### Example Usage
```
python model_training.py --input_file path/to/processed_data.csv --model_file path/to/trained_model.pkl
```
***
# PREDICT
### Overview
Is designed to streamline the process of making predictions using a pre-trained model on a given dataset. The functionality is structured into modular functions that handle loading data, loading a pre-trained model, making predictions, and saving the results in a desired format.
### Usage
1. Install Dependencies
Ensure that the required dependencies are installed. You can install them using the following command:
```
pip install pandas autogluon.tabular
```
2. Prepare Configuration
Make sure that the config.py file is properly configured. This file likely contains settings related to data processing and mapping.

3. Run the Script
Execute the script from the command line with the following arguments:

```
python model_prediction.py --input_file path/to/test_data.csv --model_file path/to/trained_model.pkl --output_file path/to/predictions.json
```
- --input_file: Path to the CSV file containing the test data.
- --model_file: Path to the pre-trained model file.
- --output_file: Path to save the predictions in JSON format.
The script will log its progress, and the final predictions will be saved in the specified output file.

### Script Structure
- load_data(file_path: str) -> pd.DataFrame
Description: Loads test data from a CSV file and returns it as a Pandas DataFrame.
- load_model(model_path: str) -> TabularPredictor
Description: Loads a pre-trained model from the specified path using the TabularPredictor.
- make_predictions(df: pd.DataFrame, model: TabularPredictor) -> dict
Description: Makes predictions using a trained model on the provided DataFrame and returns the predictions in a dictionary format.
- save_predictions(predictions, predictions_file: str)
Description: Saves predictions to a JSON file.
- parse_arguments() -> argparse.Namespace
Description: Parses command-line arguments for the model prediction script and returns an object containing the parsed arguments.
- main(input_file, model_file, output_file)
Description: Main function that orchestrates the entire prediction process. It loads data, loads the model, makes predictions, and saves the results.

Description: Parses command-line arguments and executes the main function when the script is run.
### Dependencies
- argparse: Used for parsing command-line arguments.
- json: Utilized for reading and writing JSON files.
- pandas: Essential for working with tabular data.
- autogluon: Provides the TabularPredictor class for working with tabular data and making predictions.
### Configuration
Ensure that the config.py file is properly configured with relevant settings for data processing.

### Logging
The script uses a custom logger named "Model_prediction" to log its progress. Logs are informative and include details about data loading, model loading, evaluation, and predictions.
### Example Usage
```
python model_prediction.py --input_file data/test_data.csv --model_file models/model.pkl --output_file predictions/predictions.json
```
***
For further details on each function and their parameters, refer to the function docstrings in the code.
***
REALIZADO POR Rarely Functional But Always F:
- RAMON SIEIRA MARTINEZ: www.linkedin.com/in/rasieira
- CESAR PAZ GUZMAN: www.linkedin.com/in/cesarpazguzman

