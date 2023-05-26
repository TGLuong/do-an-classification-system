import os
import re
import numpy as np
import json
from Lib.AvgSimForDetect import AvgSimForDetect
from Lib.library import *
from sklearn.neighbors import KNeighborsClassifier

class AndroidDetection:
    def __init__(self):
        self.__api_dataset = load_pkl("Res/api_dataset.pkl")
        self.__app_api = load_pkl('Res/app_api.pkl')
        self.__invoke = load_pkl('Res/invoke.pkl')
        self.__package = load_pkl('Res/package.pkl')
        self.__method = load_pkl('Res/method.pkl')
        self.__model = load_pkl('Res/model.pkl')
        

    def create_app_vector(self, data):
        self.__app_api = np.vstack([self.__app_api, app_api([data], self.__api_dataset)])
        avg_sim = AvgSimForDetect(self.__app_api, self.__invoke, self.__package, self.__method)
        feature_vector = create_vector(avg_sim, 1415, 1900, 1911, 2262, 4722)
        return feature_vector

    def predict_raw_data(self, data):
        test_feature_vector = self.create_app_vector(data)
        # print(test_feature_vector)
        predict = self.__model.predict(test_feature_vector)
        predict_proba = self.__model.predict_proba(test_feature_vector)
        result = {
            "label": predict[0],
            "probability": {
                "adware": round(predict_proba[0][0] * 100, 2),
                "banking": round(predict_proba[0][1] * 100, 2),
                "benign": round(predict_proba[0][2] * 100, 2),
                "riskware": round(predict_proba[0][3] * 100, 2),
                "smsmalware": round(predict_proba[0][4] * 100, 2)
            },
        }
        return result
    
