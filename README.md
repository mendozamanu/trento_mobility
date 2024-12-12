# trento_mobility
Experiments with a mobility dataset of the province of Trento from the Telecom Italia Big Data Challenge (2014)

*Code structure*
Once you have downloaded the telecommunications data from Trentino from Hardvard Dataverse and saved 
it in the *trento* folder (see link inside), you run processing0.py and processing1.py scripts.
This convert the data into csv format and normalizes the data.
You can modify the time period to generate the data for classification for the desired subgrid size.

The *supervised* and *clustering* folders has all the Jupyter Notebook files (.ipynb) for making the
clustering and classification.
The first notebooks that should be run are: prepare_classif_20x20 and prepare_classif_rg.
Then the clustering or kNN-day/week notebooks can be run to get the predicted saved in .csv files.

All the generated .csv files are saved in the *csv* folder and the .geojson files in the *geojson* folder.
The paint_cluster or paint_predicted* notebook files have code to create the geojson files with the 
predictions for the clustering or classification for the different grids considered.
