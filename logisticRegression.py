#%%
from turtle import width
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
import matplotlib as mpl
import datetime as dt
import time
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import csv
import joblib
import os
import config
#%%


# Class to train a Logistic Regression to classify if a cattle may be the same or not based on their velocity
class CattleClassifier:

    model = None
    def __init__(self):
        self.model = None

    # Data to train the logistic regression
    def train(self):
        with open('same_cattle1.csv', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                    s = [ float(i) for i in row ]
        f.close()


        with open('other_cattle1.csv', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                    w = [ float(i) for i in row ]
        f.close()


        with open('six_meters_velocity.csv', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                    z = [ float(i) for i in row ]
        f.close()


        #%%
        y = np.concatenate(([0]*len(s), [1]*len(w)))
        X = np.matrix(np.concatenate((s, w))).T
        X = np.asarray(X)

        #%%
        model = LogisticRegression(random_state=0)
        self.model = model.fit(X, y)

        # Persist the logistic regression model in file
        joblib.dump(self.model, config.velocity_model)

        #Score of the training
        print("Score: ")
        print(self.model.score(X, y))


        # Check if it is to plot the histogram to ilustrate the distribution of the train data
        if config.plot_histogram_when_training:

            x_list = np.arange(0, 29.1, 0.05)
            y_list_yes = []
            y_list_no = []
            x_list_yes = []
            x_list_no = []

            for x in x_list:
                pred = self.model.predict_proba([[x]])
                if pred[0][0] > 0.5:
                    y_list_yes.append(pred[0][0])
                    x_list_yes.append(x)
                else:
                    y_list_no.append(pred[0][0])
                    x_list_no.append(x)


            learned_threshold = max(x_list_yes)
            manual_threshold = 8
            print("\nLearned Threshold: " + str(learned_threshold))


            # %%
            fig, ax1 = plt.subplots(figsize=(10,8))
            histA = ax1.hist(s, bins=range(0, 30, 1), alpha=0.5, edgecolor="black", color="lightskyblue", label="Same cattle")
            histB = ax1.hist(w, bins=range(0, 30, 1), alpha=0.5, edgecolor="black", color="tomato", label="Other cattle")
            #plt.hist(z, bins=range(0, 30, 1), alpha=0.5, edgecolor="black", color="yellowgreen")
            maxY = max(histA[1].max(), histB[1].max())
            y_list_yes_norm = [(float(i)*maxY*2) for i in y_list_yes]
            y_list_no_norm = [(float(i)*maxY*2) for i in y_list_no]
            ax2 = ax1.twinx()
            ax2.plot(x_list_yes, y_list_yes, color = 'blue', linewidth=2, label="Same cattle")
            ax2.plot(x_list_no, y_list_no, color = 'red', linewidth=2, label="Other cattle")
            ax2.axvline(x=learned_threshold, color='black', linewidth=3, label="50%")
            #plt.axvline(x=manual_threshold, color='green', linewidth=3)
            ax1.set_xlabel('Velocity movement (km/hour)')
            ax1.set_ylabel('Count')
            ax2.set_ylabel('Probability')
            ax2.set_yticks(np.arange(0, 1.1, 0.2))
            plt.title("Logistic Regression for velocity attribute", fontsize=20)
            plt.legend()
            ax1.set_xlim([0, 29])
            list = np.append(np.arange(0, 30, 2), (round(learned_threshold, 0)))
            list = np.sort(list)
            plt.xticks(list)
            plt.show()

            # %%
            print(self.model.coef_)
            print(self.model.intercept_)

    def load_model(self):
        self.model = joblib.load(config.velocity_model)


    # Given a velocity, method predict the probability of the cattle be the same
    def predict(self, velocity):
        if self.model is None:
            self.load_model()
        proba = self.model.predict_proba([[velocity]])[0][1]
        return proba

