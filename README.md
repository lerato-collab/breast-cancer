# Breast Cancer Prediction

This project predicts whether a breast tumour is malignant or benign using the Wisconsin Breast Cancer dataset from Kaggle. The dataset has measurements taken from digitized images of fine needle aspirates, such as the radius, texture, perimeter and area of the cell nuclei. I built it as a Kaggle notebook to practise a full machine learning workflow, from exploring the data to comparing models.

## Dataset

The data is a file called `data.csv` from the Breast Cancer dataset on Kaggle. It has 569 samples and 30 numeric features. 357 of the samples are benign (62.7%) and 212 are malignant (37.3%), so the classes are fairly balanced but there are more benign cases.

On Kaggle the file is found at `/kaggle/input/breast-cancer-dataset/data.csv`.

## What I did

I started by loading the data and checking its shape, the class balance and any missing values. After that I looked at how the features are distributed, how they correlate with each other, and whether there were outliers. Many of the features are strongly correlated, which makes sense because things like radius, perimeter and area all describe the size of the nucleus.

For preprocessing I encoded the diagnosis as a number, scaled the features and split the data into training and test sets.

I then trained three models and compared them: logistic regression, random forest and a support vector machine. I checked each one with accuracy and the ROC curve, and looked at feature importance from the random forest. I also ran PCA to see how much the number of features could be reduced.

## Results

All three models classified the tumours well on the test set. Logistic regression reached 96.5% accuracy with an AUC of 0.996. Random forest and the SVM both reached 97.4% accuracy, with AUC values of 0.993 and 0.995. Random forest came out as the best model in my notebook, although the SVM scored the same accuracy.

The five most important features according to the random forest were area_worst, concave points_worst, radius_worst, perimeter_worst and concave points_mean. The "worst" measurements seem to carry the most information about whether a tumour is malignant.

The PCA showed that 10 components are enough to keep 95% of the variance, and the first two components explain 63.1% of it. Benign and malignant samples also form fairly clear clusters when plotted on those first two components.

## Limitations

I used a single train and test split, so these scores could change a little with a different split. Cross-validation would give a more reliable estimate and is something I would add next. The dataset is also small and clean, which is part of why every model scores so highly. This project is for learning and is not a tool for real medical diagnosis.

## Screenshots

The notebook code lives on Kaggle. This repository holds screenshots of the notebook and its outputs, which you can find in the folder [screenshot for the breast cancer project](<./screenshot for the breast cancer project>).

## Running it yourself

To try it on Kaggle, create a new notebook, add the Breast Cancer dataset to it, and run the cells in order. You can check that the data is attached with this cell:

```python
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```

## Author

Lerato Nkhuishe  
GitHub: [lerato-collab](https://github.com/lerato-collab)  
Kaggle: [leratocollab](https://www.kaggle.com/leratocollab)

## Acknowledgements

The dataset comes from the UCI Machine Learning Repository and the Wisconsin Breast Cancer Database, shared on Kaggle.
