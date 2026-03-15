from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).parent

app = FastAPI()

origins = [
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "https://disease-prediction-frontend-ejut.onrender.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class model_input(BaseModel):
   abdominal_pain : float
   abnormal_menstruation : float
   acidity : float
   acute_liver_failure : float
   altered_sensorium : float
   anxiety : float
   back_pain : float
   belly_pain : float
   blackheads : float
   bladder_discomfort : float
   blister : float
   blood_in_sputum : float
   bloody_stool : float
   blurred_and_distorted_vision : float
   breathlessness : float
   brittle_nails : float
   bruising : float
   burning_micturition : float
   chest_pain : float
   chills : float
   cold_hands_and_feets : float
   coma : float
   congestion : float
   constipation : float
   continuous_feel_of_urine : float
   continuous_sneezing : float
   cough : float
   cramps : float
   dark_urine : float
   dehydration : float
   depression : float
   diarrhoea : float
   #
   dischromic_patches : float
   #
   distention_of_abdomen : float
   dizziness : float
   drying_and_tingling_lips : float
   enlarged_thyroid : float
   excessive_hunger : float
   extra_marital_contacts : float
   family_history : float
   fast_heart_rate : float
   fatigue : float
   fluid_overload : float
   #
   foul_smell_ofurine : float
   #
   headache : float
   high_fever : float
   hip_joint_pain : float
   history_of_alcohol_consumption : float
   increased_appetite : float
   indigestion : float
   inflammatory_nails : float
   internal_itching : float
   irregular_sugar_level : float
   irritability : float
   irritation_in_anus : float
   joint_pain : float
   knee_pain : float
   lack_of_concentration : float
   lethargy : float
   loss_of_appetite : float
   loss_of_balance : float
   loss_of_smell : float
   malaise : float
   mild_fever : float
   mood_swings : float
   movement_stiffness : float
   mucoid_sputum : float
   muscle_pain : float
   muscle_wasting : float
   muscle_weakness : float
   nausea : float
   neck_pain : float
   nodal_skin_eruptions : float
   obesity : float
   pain_behind_the_eyes : float
   pain_during_bowel_movements : float
   pain_in_anal_region : float
   painful_walking : float
   palpitations : float
   passage_of_gases : float
   patches_in_throat : float
   phlegm : float
   polyuria : float
   prominent_veins_on_calf : float
   puffy_face_and_eyes : float
   pus_filled_pimples : float
   receiving_blood_transfusion : float
   receiving_unsterile_injections : float
   red_sore_around_nose : float
   red_spots_over_body : float
   redness_of_eyes : float
   restlessness : float
   runny_nose : float
   rusty_sputum : float
   scurring : float
   shivering : float
   silver_like_dusting : float
   sinus_pressure : float
   skin_peeling : float
   skin_rash : float
   slurred_speech : float
   small_dents_in_nails : float
   spinning_movements : float
   #
   spotting_urination : float
   #
   stiff_neck : float
   stomach_bleeding : float
   stomach_pain : float
   sunken_eyes : float
   sweating : float
   swelled_lymph_nodes : float
   swelling_joints : float
   swelling_of_stomach : float
   swollen_blood_vessels : float
   swollen_extremeties : float
   swollen_legs : float
   throat_irritation : float
   #
   toxic_look_typhos : float
   #
   ulcers_on_tongue : float
   unsteadiness : float
   visual_disturbances : float
   vomiting : float
   watering_from_eyes : float
   weakness_in_limbs : float
   weakness_of_one_body_side : float
   weight_gain : float
   weight_loss : float
   yellow_crust_ooze : float
   yellow_urine : float
   yellowing_of_eyes : float
   yellowish_skin : float
   blank : float
   itching : float

    # class Config:
    #     arbitrary_types_allowed = True

with open(HERE / "dp_model",'rb') as f:
    pred_model = pickle.load(f)
# pred_model = pickle.load(open(HERE / "dp_model"),'rb')
# C:\Users\Harshit Bajpai\Desktop\Disease_prediciton\dp_pred\dp_model.sav


@app.post('/dp_pred')
def dp_predictor(input_parameters:model_input):
    input_data = input_parameters.model_dump_json()
    input_dictionary = json.loads(input_data)
    # print("Input Dictionary :", input_dictionary)
    input_list = list(input_dictionary.values())
    # input_list = converter(input_dictionary)
    print("input values are = ", input_list)
        # input_list2 = [33, 35, 35, 50, 38, 32, 26, 21, 22, 21, 18, 11, 8, 4, 3, 3, 1]
    prediction = pred_model.predict([input_list])
    # print("Result of pred: ", prediction)
    pred_list = list(prediction)
    pred_json = json.dumps(pred_list)
    # print("Result of pred in json: ", pred_json)
    return pred_json