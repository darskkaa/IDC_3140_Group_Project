# car rental price prediction
# IDC 3140 - group project

import subprocess
subprocess.run(['pip', 'install', 'kagglehub', '-q'], check=True)

import kagglehub
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier

USD_RATE = 300
