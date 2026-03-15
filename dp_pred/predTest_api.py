from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import warnings
from warnings import filterwarnings
filterwarnings("ignore")
print(pickle.format_version)

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

    Belly_button_that_sticks_out: int
    Bulge_in_the_groin_or_scrotum: int
    Delayed_sexual_maturity: int
    Delayed_teeth: int
    Downward_palpebral_slant_to_eyes: int
    Hairline_with_a_widows_peak: int
    Mildly_sunken_chest_pectus_excavatum: int
    Mild_to_moderate_cognitive_problems: int
    Mild_to_moderate_short_height: int
    Poorly_developed_middle_section_of_the_face: int
    Short_fingers_and_toes_with_mild_webbing: int
    Single_crease_in_the_palm_of_the_hand: int
    Small_nose_with_nostrils_tipped_forward: int
    Testicles_that_have_not_come_down_undescended: int
    Top_portion_of_the_ear_folded_over_slightly: int
    Wide_set_eyes_with_droopy_eyelids: int
    Absent_or_small_knuckles: int
    Cleft_palate: int
    Deformed_ears: int
    Inability_to_fully_extend_the_joints_from_birth: int
    Narrow_shoulders: int
    Pale_skin: int
    Triple_jointed_thumbs: int
    Pain_in_the_abdomen: int
    Fainting_Drowsiness: int
    Clammy_skin: int
    Nausea_and_vomiting: int
    Rapid_heart_rate: int
    Shock: int
    Bleeding_or_spotting_from_the_vagina_between_periods: int
    Periods_that_occur_less_than_28_days_apart_or_more_than_35_days_apart: int
    Time_between_periods_changes_each_month: int
    Heavier_bleeding: int
    Bleeding_that_lasts_for_more_days_than_normal_or_for_more_than_7_days: int
    Tenderness_and_dryness_of_the_vagina: int
    Hot_flashes: int
    Mood_swings: int
    Shallow_breathing: int
    Slow_and_labored_breathing: int
    Stopped_breathing_or_respiratory_arrest: int
    Very_small_pupils: int
    Low_blood_pressure: int
    Coma_lack_of_responsiveness: int
    Convulsions_seizures: int
    Stupor_lack_of_alertness: int
    Bluish_skin_fingernails_and_lips: int
    Spasms_of_the_stomach_and_intestines: int
    Liver_failure: int
    Kidney_failure: int
    Person_may_have_a_fruity_odor: int
    Sweet_taste_in_mouth: int
    Feeling_of_drunkenness: int
    Lack_of_coordination: int
    Difficulty_breathing: int
    Increased_need_to_urinate: int
    Backflow_regurgitation_of_food: int
    Chest_pain_which_may_increase_after_eating_or_may_be_felt_as_pain_in_the_back_neck_and_arms: int
    Cough: int
    Difficulty_swallowing_liquids_and_solids: int
    Heartburn: int
    Unintentional_weight_loss: int
    Very_short_trunk_arms_legs_and_neck: int
    Head_appears_large_in_relation_to_the_trunk: int
    Small_lower_jaw: int
    Narrow_chest: int
    Unusually_large_head: int
    Large_forehead_and_flat_bridge_of_the_nose: int
    Crowded_or_crooked_teeth: int
    Short_stature_well_below_the_average_height_for_a_person_of_the_same_age_and_sex: int
    Average_size_trunk_with_short_arms_and_legs_especially_the_upper_arms_and_thighs: int
    Bowed_legs: int
    Limited_range_of_motion_of_the_elbows: int
    Spine_curvatures_called_kyphosis_and_lordosis: int
    Short_fingers_with_an_extra_space_between_the_ring_and_middle_finger_trident_hand: int
    Decreased_muscle_tone_in_infants: int
    Loss_of_vision: int
    Severe_pain_in_the_throat: int
    Severe_pain_or_burning_in_the_nose_eyes_ears_lips_or_tongue: int
    Decreased_urine_output: int
    Blood_in_the_stool: int
    Burns_of_the_food_pipe_esophagus: int
    Vomiting_blood: int
    Irregular_heart_beat: int
    Throat_swelling: int
    Holes_in_the_skin_or_tissues_under_the_skin: int
    Confusion: int
    Fatigue_feeling_tired: int
    Lethargy: int
    Shortness_of_breath: int
    Sleepiness: int
    Crusting_of_skin_bumps: int
    Cysts: int
    Papules_small_red_bumps: int
    Pustules_small_red_bumps_containing_white_or_yellow_pus: int
    Redness_around_the_skin_eruptions: int
    Scarring_of_the_skin: int
    Blackheads: int
    Abnormal_feeling_of_movement: int
    Hearing_loss: int
    Ringing_tinnitus_in_the_affected_ear: int
    Difficulty_understanding_speech: int
    Headache: int
    Loss_of_balance: int
    Numbness_in_the_face_or_one_ear: int
    Pain_in_the_face_or_one_ear: int
    Weakness_of_the_face_or_facial_asymmetry: int
    Frequent_middle_ear_infections: int
    Growth_problems_short_arms_and_legs: int
    Hearing_problems: int
    Intellectual_disability: int
    The_body_doesnt_respond_to_certain_hormones_even_though_hormone_levels_are_normal: int
    Distinct_facial_features: int
    Body_odor: int
    Carpal_tunnel_syndrome: int
    weakness: int
    Decreased_peripheral_vision: int
    Gaps_between_the_teeth: int
    Enlarged_tongue: int
    Excessive_height: int
    Excessive_sweating: int
    Heart_enlargement_which_can_cause_fainting_or_shortness_of_breath: int
    Hoarseness: int
    Jaw_pain: int
    Joint_pain: int
    Large_bones_of_the_face_large_jaw_and_tongue_widely_spaced_teeth: int
    Large_feet_change_in_shoe_size_large_hands_change_in_ring_or_glove_size: int
    Large_glands_in_the_skin_sebaceous_glands_causing_oily_skin_thickening_of_the_skin_skin_tags_growths: int
    Sleep_apnea: int
    Dehydration: int
    Fever: int
    Loss_of_appetite: int
    Low_blood_sugar: int
    Rapid_respiratory_rate: int
    Slow_sluggish_movement: int
    Unusual_and_excessive_sweating_on_face_or_palms: int
    Back_pain: int
    Blood_in_the_urine: int
    Pain_in_the_side: int
    Chest_discomfort: int
    Cough_that_produces_mucus_the_mucus_may_be_clear_or_yellow_green: int
    Fever_usually_low_grade: int
    Shortness_of_breath_that_gets_worse_with_activity: int
    Wheezing: int
    Clumsy_speech_pattern_dysarthria: int
    Repetitive_eye_movements_nystagmus: int
    Uncoordinated_eye_movements: int
    Walking_problems_unsteady_gait_that_can_lead_to_falls: int
    Difficulty_controlling_arm_movements: int
    Sharp_cramping_or_dull_pain: int
    Steady_pain: int
    Pain_that_spreads_to_your_back_or_below_your_right_shoulder_blade: int
    Clay_colored_stools: int
    Yellowing_of_skin_and_whites_of_the_eyes_jaundice: int
    Pain_in_the_shoulder_arm_neck_jaw_back_or_belly_area: int
    Discomfort_that_feels_like_tightness_squeezing_crushing_burning_choking_or_aching: int
    Discomfort_that_occurs_at_rest_and_does_not_easily_go_away_when_you_take_medicine: int
    Anxiety: int
    Feeling_dizzy_or_lightheaded: int
    Fast_or_irregular_heartbeat: int
    Drooping_eyelids: int
    Difficulty_moving_the_eyes: int
    Slurred_speech_or_difficulty_swallowing: int
    Stiffness_in_the_neck: int
    Pain_in_the_arms_or_legs: int
    Inability_to_pass_urine: int
    Respiratory_failure_when_muscles_involved_in_breathing_become_weak: int
    Serious_nervous_system_problems_which_may_lead_to_death: int
    Breath_odor_and_metallic_taste_in_the_mouth: int
    Changes_in_mental_status_or_mood: int
    Decreased_sensation_especially_in_the_hands_or_feet: int
    Flank_pain_between_the_ribs_and_hips: int
    Hand_tremor: int
    Heart_murmur: int
    High_blood_pressure: int
    Nosebleeds: int
    Persistent_hiccups: int
    Prolonged_bleeding: int
    Swelling_due_to_the_body_keeping_in_fluid_may_be_seen_in_the_legs_ankles_and_feet: int
    Urination_changes_such_as_little_or_no_urine_excessive_urination_at_night_or_urination_that_stops_completely: int
    Bone_and_joint_pain: int
    Easy_bruising_and_bleeding: int
    Paleness: int
    Pain_or_feeling_of_fullness_below_the_ribs_from_an_enlarged_liver_or_spleen: int
    Pinpoint_red_spots_on_the_skin: int
    Swollen_lymph_nodes_in_the_neck_under_arms_and_groin: int
    Night_sweats: int
    Difficulty_sleeping: int
    Chest_tightness_or_congestion: int
    Inability_to_walk_in_a_straight_line_or_walk_at_all: int
    Bleeding_and_swelling_rare_in_the_gums: int
    Bone_pain_or_tenderness: int
    Heavy_menstrual_periods: int
    Swelling_of_the_face: int
    Swelling_of_eye: int
    Swelling_of_legs: int
    Swelling_of_arms: int
    Swelling_of_hands: int
    Swelling_of_feet: int
    Swelling_of_abdomen: int
    Bloating_or_abdominal_gas: int
    Indigestion: int
    General_swelling: int
    Diarrhea: int
    Darkening_of_the_skin: int
    Salt_craving: int
    Heavy_menstrual_bleeding: int
    Painful_menstrual_periods: int
    Pelvic_pain_during_intercourse: int
    Constipation: int
    No_longer_being_able_to_pass_gas: int
    Acting_defiant_or_showing_impulsive_behavior: int
    Acting_nervous: int
    Crying_feeling_sad_or_hopeless_and_possibly_withdrawing_from_other_people: int
    Skipped_heartbeats_and_other_physical_complaints: int
    Trembling_or_twitching: int
    Changes_in_muscle_tone_especially_muscle_spasms_and_uncontrolled_movements: int
    Crossed_eyes: int
    Handwriting_that_gets_worse: int
    Difficulty_at_school: int
    Difficulty_understanding_what_people_are_saying: int
    Hyperactivity: int
    Visual_impairment_or_blindness: int
    Bronchitis: int
    Conjunctivitis: int
    Otitis_media_middle_ear_infection: int
    Pneumonia_lung_infection: int
    Sinusitis_sinus_infection: int
    Skin_infections: int
    Upper_respiratory_tract_infections: int
    Being_afraid_of_spending_time_alone: int
    Being_afraid_of_places_where_escape_might_be_hard: int
    Being_afraid_of_losing_control_in_a_public_place: int
    Depending_on_others: int
    Feeling_detached_or_separated_from_others: int
    Feeling_helpless: int
    Feeling_that_the_body_is_not_real: int
    Feeling_that_the_environment_is_not_real: int
    Having_an_unusual_temper_or_agitation: int
    Staying_in_the_house_for_long_periods: int
    Malaise_general_discomfortness_ill_feeling_or_uneasiness: int
    Chills: int
    Sore_throat: int
    Mouth_and_throat_ulcers: int
    No_color_in_the_hair_skin_or_iris_of_the_eye: int
    Lighter_than_normal_skin_and_hair: int
    Patches_of_missing_skin_color: int
    Light_sensitivity: int
    Rapid_eye_movements: int
    Vision_problems_or_functional_blindness: int
    Lightheadedness: int
    Muscle_twitching: int
    Prolonged_muscle_spasms_tetany: int
    Intense_itching_or_burning_eyes: int
    Puffy_eyelids_most_often_in_the_morning: int
    Red_eyes: int
    Stringy_eye_discharge: int
    Tearing_watery_eyes: int
    Widened_blood_vessels_in_the_clear_tissue_covering_the_white_of_the_eye: int
    Hives_especially_over_the_neck_and_face: int
    Itching: int
    Nasal_congestion: int
    Rashes: int
    Problems_with_smell: int
    Runny_nose: int
    Sneezing: int
    Abnormal_urine_color: int
    Excessive_thirst: int
    Abnormal_shape_of_the_lens: int
    Corneal_erosion: int
    Abnormal_coloring_of_the_retina: int
    Macular_hole: int
    Deafness: int
    Impaired_heart_function: int
    Obesity: int
    Progressive_kidney_failure: int
    Slowed_growth: int
    Type_2_diabetes: int
    Difficulty_performing_more_than_one_task_at_a_time: int
    Getting_lost_on_familiar_routes: int
    Language_problems_such_as_trouble_remembering_the_names_of_familiar_objects: int
    Losing_interest_in_thing_previously_enjoyed_and_being_in_a_flat_mood: int
    Misplacing_items: int
    Personality_changes_and_loss_of_social_skills: int
    Change_in_sleep_patterns_often_waking_up_at_night: int
    Delusions_depression_and_agitation: int
    Difficulty_doing_basic_tasks: int
    Difficulty_reading_or_writing: int
    Forgetting_details_about_current_events: int
    Hallucinations: int
    Abdominal_cramps: int
    Excessive_gas: int
    Rectal_pain_while_having_a_bowel_movement_tenesmus: int
    Abdominal_pain_more_so_in_the_right_upper_part_of_the_abdomen_pain_is_intense_continuous_or_stabbing: int
    Difficulty_lifting_climbing_stairs_and_walking: int
    Difficulty_swallowing_choking_easily_drooling_or_gagging: int
    Head_drop_due_to_weakness_of_the_neck_muscles: int
    Speech_problems_such_as_a_slow_or_abnormal_speech_pattern_slurring_of_words: int
    Muscle_cramps: int
    Muscle_stiffness: int
    Muscle_contractions_called_fasciculations: int
    A_lump_in_or_near_the_anus: int
    Anal_pain: int
    Itching_in_anal: int
    Discharge_from_the_anus: int
    Change_in_bowel_habits: int
    Swollen_lymph_nodes_in_the_groin_or_anal_region: int
    Decreased_alertness: int
    Feeling_anxious: int
    Palpitations: int
    Swelling_of_tongue: int
    Unconsciousness: int
    Coughing_up_blood: int
    Loud_breathing: int
    Lower_neck_lump_which_often_grows_quickly: int
    Pain_in_throat: int
    Vocal_cord_paralysis: int
    Overactive_thyroid: int
    Feeling_weak_or_tired_more_often_than_usual_or_with_exercise: int
    Irritability: int
    Numbness_and_tingling_of_hands_or_feet: int
    Blue_color_to_the_whites_of_the_eyes: int
    Brittle_nails: int
    Desire_to_eat_ice_or_other_non_food_things_pica_syndrome: int
    Sore_or_inflamed_tongue: int
    Abnormal_or_increased_menstrual_bleeding_in_females: int
    Loss_of_sexual_desire: int
    Loss_of_muscle_tone_floppiness: int
    Trouble_feeding: int
    Trembling_arm_and_leg_movements: int
    Unstable_or_jerky_walking: int
    Little_or_no_speech: int
    Laughing_and_smiling_often: int
    Light_hair_skin_and_eye_color_compared_to_rest_of_family: int
    Small_head_size_compared_to_body_flattened_back_of_head: int
    Excessive_movement_of_the_hands_and_limbs: int
    Sleep_problems: int
    Tongue_thrusting_drooling: int
    Unusual_chewing_and_mouthing_movements: int
    Walking_with_arms_uplifted_and_hands_waving: int
    Discharge_of_pus_from_the_rectum: int
    Redness_painful_and_hardened_tissue_in_the_area_of_the_anus: int
    Tenderness_In_anus_swollen_part: int
    Intense_fear_of_gaining_weight_or_becoming_fat: int
    Refuses_to_keep_their_weight_at_what_is_considered_normal_for_their_age_and_height_15_or_more_below_the_normal_weight: int
    Has_a_body_image_that_is_very_distorted_is_very_focused_on_body_weight_or_shape_and_refuse_to_admit_the_danger_of_weight_loss: int
    Cutting_food_into_small_pieces_or_moving_them_around_the_plate_instead_of_eating: int
    Exercising_all_the_time_even_when_the_weather_is_bad_they_are_hurt_or_their_schedule_is_busy: int
    Going_to_the_bathroom_right_after_meals: int
    Refusing_to_eat_around_other_people: int
    Using_pills_to_make_themselves_urinate: int
    A_popping_sound_at_the_time_of_injury: int
    Obvious_knee_swelling_within_6_hours_of_injury: int
    Pain_especially_when_you_try_to_put_weight_on_the_injured_leg: int
    Difficulty_in_continuing_with_your_sport: int
    Feeling_of_instability: int
    An_itchy_sore_develops_that_is_similar_to_an_insect_bite: int
    The_sore_is_usually_painless: int
    A_scab_often_forms_and_then_dries_and_falls_off_within_2_weeks: int
    Chest_pain: int
    Burning_sensation_when_urinating: int
    Swelling_edema_in_any_area_of_the_body_especially_in_the_legs: int
    Vomiting: int
    Collapse_of_heart_beat: int
    Methemoglobinemia_very_dark_blood_from_abnormal_red_blood_cells: int
    Too_much_or_too_little_acid_in_the_blood_which_leads_to_damage_in_all_of_the_body_organs: int
    Asphyxia: int
    Chemical_pneumonitis: int
    Viral_infection: int
    Hemorrhagic_pulmonary_edema: int
    Respiratory_distress_or_failure: int
    Pneumothorax: int
    Pleural_effusion: int
    Empyema: int
    Incoordination: int
    Somnolence: int
    Blurred_vision: int
    Brain_damage_from_low_oxygen_level: int
    Burns_in_skin: int
    Irritation_in_skin: int
    Holes_necrosis_in_the_skin_or_tissues_underneath: int
    Bruising_and_bleeding_under_the_skin: int
    Apathy_loss_of_desire_to_do_anything: int
    Slowdown_or_stoppage_of_the_bowels: int
    Dry_mucous_membranes_in_the_mouth: int
    Eye_changes_in_pupil_size: int
    Eyes_move_quickly_from_side_to_side: int
    Flushed_skin: int
    Restlessness: int
    Urination_difficulty: int
    Leg_cramps: int
    Unsteady_walk: int
    Blue_lips_and_fingernails: int
    Act_witty_and_charming: int
    Good_at_flattery_and_manipulating_other_peoples_emotions: int
    Break_the_law_repeatedly: int
    Disregard_the_safety_of_self_and_others: int
    Have_problems_with_substance_abuse: int
    Lie_steal_and_fight_often: int
    Not_show_guilt_or_remorse: int
    Often_be_angry_or_arrogant: int
    Blood_pressure_changes: int
    Reduced_pulse: int
    Transient_ischemic_attacks_TIA: int
    Sharp_stabbing_tearing_or_ripping_chest_pain: int
    Anxiety_and_a_feeling_of_doom: int
    Stroke_symptoms: int
    Bounding_pulse: int
    Waking_up_short_of_breath_some_time_after_falling_asleep: int
    Uneven_rapid_racing_pounding_or_fluttering_pulse: int
    Breathing_problems_when_exercising: int
    Becoming_easily_tired: int
    Failure_to_gain_weight: int
    Poor_feeding: int
    Heart_failure: int
    Respiratory_infections: int
    Early_closure_of_sutures_between_bones_of_the_skull_noted_by_ridging_along_sutures: int
    Frequent_ear_infections: int
    Fusion_or_severe_webbing_of_the_2nd_3rd_and_4th_fingers_often_called_mitten_hands: int
    Large_or_late_closing_soft_spot_on_a_babys_skull: int
    Possible_slow_intellectual_development: int
    Prominent_or_bulging_eyes: int
    Severe_under_development_of_the_midface: int
    Skeletal_limb_abnormalities: int
    Webbing_or_fusion_of_the_toes: int
    Low_white_cell_count_leukopenia: int
    Low_platelet_count_thrombocytopenia: int
    Bleeding_gums: int
    Rash_small_pinpoint_red_marks_on_the_skin_petechiae: int
    Severe_infections: int
    Pain_may_be_worse_when_you_walk_cough_or_make_sudden_movements: int
    Chills_and_shaking: int
    Hard_stools: int
    Distorted_repeated_or_left_out_speech_sounds_or_words_the_person_has_difficulty_putting_words_together_in_the_correct_order: int
    Struggling_to_pronounce_the_right_word: int
    More_difficulty_using_longer_words_either_all_the_time_or_sometimes: int
    Ability_to_use_short_everyday_phrases_or_sayings_such_as_How_are_you_without_a_problem: int
    Better_writing_ability_than_speaking_ability: int
    Too_slow_bradycardia_heart_rate: int
    Too_quick_tachycardia_heart_rate: int
    Irregular_uneven_possibly_with_extra_or_skipped_heart_beats: int
    Cold_arm_or_leg: int
    Decreased_or_no_pulse_in_an_arm_or_leg: int
    Lack_of_movement_in_the_arm_or_leg: int
    Pale_color_of_the_arm_or_leg_pallor: int
    Weakness_of_an_arm_or_leg: int
    Joint_swelling: int
    Reduced_ability_to_move_the_joint: int
    Redness_and_warmth_of_the_skin_around_a_joint: int
    Joint_stiffness: int
    Clubbing_of_fingers: int
    Nail_abnormalities: int
    Bloody_sputum: int
    Passing_worms_in_stool: int
    Coughing_up_worms: int
    Worms_leaving_the_body_through_the_nose_or_mouth: int
    Skin_sores_lesions: int
    Coughing_up_foul_smelling_greenish_or_dark_phlegm_sputum_or_phlegm_that_contains_pus_or_blood: int
    Seeing_food_or_tube_feed_material_if_being_fed_artificially_in_your_sputum: int
    Cough_with_or_without_sputum_phlegm_production: int
    Pulling_in_of_the_skin_between_the_ribs_when_breathing_intercostal_retractions: int
    Abnormal_breathing_pattern_breathing_out_takes_more_than_twice_as_long_as_breathing_in: int
    Bluish_color_to_the_lips_and_face_cyanosis: int
    Decreased_coordination_of_movements_ataxia_in_late_childhood_that_can_include_ataxic_gait_cerebellar_ataxia_jerky_gait_unsteadiness: int
    Decreasing_mental_development_slows_or_stops_after_age_10_to_12: int
    Delayed_walking: int
    Discoloration_of_skin_areas_exposed_to_sunlight: int
    Discoloration_of_skin_coffee_with_milk_colored_spots: int
    Enlarged_blood_vessels_in_skin_of_nose_ears_and_inside_of_the_elbow_and_knee: int
    Enlarged_blood_vessels_in_the_whites_of_the_eyes: int
    Jerky_or_abnormal_eye_movements_nystagmus_late_in_the_disease: int
    Premature_graying_of_the_hair: int
    Sensitivity_to_radiation_including_x_rays: int
    Severe_respiratory_infections_that_keep_coming_back_recurring: int
    Blisters_with_oozing_and_crusting: int
    Dry_skin_all_over_the_body_or_areas_of_bumpy_skin_on_the_back_of_the_arms_and_front_of_the_thighs: int
    Ear_discharge_or_bleeding: int
    Raw_areas_of_the_skin_from_scratching: int
    Skin_color_changes_such_as_more_or_less_color_than_the_normal_skin_tone: int
    Skin_redness: int
    Thickened_or_leather_like_areas_which_can_occur_after_long_term_irritation_and_scratching: int
    Loss_of_ability_to_exercise: int
    Breathing_difficulty_when_lying_flat: int
    Breathing_difficulty_when_asleep: int
    Symptoms_due_to_embolism_of_tumor_material: int
    Bluish_skin_especially_on_the_fingers_Raynaud_phenomenon: int
    Curvature_of_nails_accompanied_by_soft_tissue_swelling_clubbing_of_the_fingers: int
    Fingers_that_change_color_upon_pressure_or_with_cold_or_stress: int
    Doesnt_pay_attention_to_details_or_makes_careless_mistakes: int
    Has_problems_focusing_during_tasks_or_play: int
    Doesnt_listen_when_spoken_to_directly: int
    Doesnt_follow_through_on_instructions_and_doesnt_finish_schoolwork_or_chores: int
    Has_problems_organizing_tasks_and_activities: int
    Avoids_or_doesnt_like_tasks_that_require_mental_effort: int
    Often_loses_things: int
    Easily_distracted: int
    Often_forgetful: int
    Fidgets_or_squirms_in_seat: int
    Leaves_their_seat_when_they_should_stay_in_their_seat: int
    Runs_about_or_climbs_when_they_shouldnt_be_doing_so: int
    Has_problems_playing_or_working_quietly: int
    Is_often_on_the_go_acts_as_if_driven_by_a_motor: int
    Talks_all_the_time: int
    Blurts_out_answers_before_questions_have_been_completed: int
    Has_problems_awaiting_their_turn: int
    Interrupts_or_intrudes_on_others: int
    Be_very_sensitive_to_sight_hearing_touch_smell_or_taste: int
    Be_very_upset_when_routines_are_changed: int
    Repeat_body_movements_over_and_over: int
    Be_unusually_attached_to_things: int
    Cant_start_or_maintain_a_conversation: int
    Uses_gestures_instead_of_words: int
    Develops_language_slowly_or_not_at_all: int
    Doesnt_adjust_gaze_to_look_at_objects_that_others_are_looking_at: int
    Doesnt_refer_to_self_the_right_way: int
    Doesnt_point_to_show_other_people_objects: int
    Repeats_words_or_memorized_passages_such_as_commercials: int
    Doesnt_make_friends: int
    Doesnt_play_interactive_games: int
    Is_withdrawn_from_social_activity: int
    May_not_respond_to_eye_contact_or_smiles_or_may_avoid_eye_contact: int
    May_treat_others_as_objects: int
    Prefers_to_be_alone_rather_than_with_others: int
    Isnt_able_to_show_empathy: int
    Doesnt_startle_at_loud_noises: int
    Has_very_high_or_very_low_senses_of_sight_hearing_touch_smell_or_taste: int
    May_find_normal_noises_painful_and_hold_their_hands_over_their_ears: int
    May_withdraw_from_physical_contact_because_its_too_stimulating_or_overwhelming: int
    Rubs_surfaces_mouths_or_licks_objects: int
    May_have_a_very_high_or_very_low_response_to_pain: int
    Doesnt_imitate_the_actions_of_others: int
    Prefers_solitary_or_ritualistic_play: int
    Shows_little_pretend_or_imaginative_play: int
    Acts_out_with_intense_tantrums: int
    Gets_stuck_on_a_single_topic_or_task: int
    Has_a_short_attention_span: int
    Has_very_narrow_interests: int
    Is_overactive_or_very_passive: int
    Is_aggressive_toward_others_or_self: int
    Shows_a_strong_need_for_things_being_the_same: int
    Repeats_body_movements: int
    Abdominal_distention: int
    Bladder_or_bowel_problems: int
    Goosebumps_flushed_red_skin_above_the_level_of_the_spinal_cord_injury: int
    Irregular_heartbeat_slow_or_fast_pulse: int
    Muscle_spasms: int
    Throbbing_headache: int
    Feeling_full_after_only_a_few_bites: int
    Problems_controlling_bowel_movements: int
    Blood_pressure_drop_with_position_that_causes_dizziness_when_standing: int
    Difficulty_beginning_to_urinate: int
    Feeling_of_incomplete_bladder_emptying: int
    Leaking_urine: int
    Easily_hurt_when_people_criticize_or_disapprove_of_them: int
    Hold_back_too_much_in_intimate_relationships: int
    Reluctant_to_become_involved_with_people: int
    Avoid_activities_or_jobs_that_involve_contact_with_others: int
    Be_shy_in_social_situations_out_of_fear_of_doing_something_wrong: int
    Make_potential_difficulties_seem_worse_than_they_are: int
    Hold_the_view_they_are_not_good_socially_not_as_good_as_other_people_or_unappealing: int
    Redness_of_foreskin_or_penis: int
    Other_rashes_on_the_head_of_the_penis: int
    Foul_smelling_discharge: int
    Painful_penis_and_foreskin: int
    Altered_level_of_consciousness: int
    Difficulty_in_thinking: int
    Faulty_judgment: int
    Sluggishness: int
    Staggering_difficulty_with_balance: int
    A_tender_lump_on_either_side_of_the_vaginal_opening: int
    Swelling_and_redness: int
    Pain_with_sitting_or_walking: int
    Pain_with_sexual_intercourse: int
    Vaginal_discharge: int
    Vaginal_pressure: int
    Rate_of_weight_gain_is_much_lower_than_that_of_other_children_of_similar_age_and_sex: int
    Kidney_stones: int
    Movement_changes_such_as_involuntary_or_slowed_movements: int
    Increased_muscle_tone: int
    Tremor_in_full_body: int
    Problems_finding_words: int
    Uncontrollable_repeated_movements_speech_or_cries_tics: int
    Walking_difficulty: int
    Curvature_of_spine: int
    Poor_muscle_coordination_that_usually_develops_after_age_10: int
    Protruding_abdomen: int
    Stool_abnormalities_including_fatty_stools_that_appear_pale_in_color_frothy_stools_and_abnormally_foul_smelling_stools: int
    Difficulty_walking_that_gets_worse_over_time_by_age_25_to_30_the_person_is_usually_unable_to_walk: int
    Frequent_falls: int
    Difficulty_getting_up_from_the_floor_and_climbing_stairs: int
    Difficulty_with_running_hopping_and_jumping: int
    Loss_of_muscle_mass: int
    Toe_walking: int
    Muscle_weakness_in_the_arms_neck_and_other_areas_is_not_as_severe_as_in_the_lower_body: int
    Large_size_for_a_newborn: int
    Red_birth_mark_on_forehead_or_eyelids_nevus_flammeus: int
    Creases_in_ear_lobes: int
    Large_tongue_macroglossia: int
    Abdominal_wall_defect: int
    Enlargement_of_some_organ: int
    Overgrowth_of_one_side_of_the_body_hemihyperplasia_hemihypertrophy: int
    Tumor_growth_such_as_Wilms_tumors_and_hepatoblastomas: int
    Difficulty_closing_one_eye: int
    Difficulty_eating_and_drinking_because_food_falls_out_of_one_side_of_the_mouth: int
    Drooling_due_to_lack_of_control_over_the_muscles_of_the_face: int
    Drooping_of_the_face_such_as_the_eyelid_or_corner_of_the_mouth: int
    Problems_smiling_grimacing_or_making_facial_expressions: int
    Dry_eye_which_may_lead_to_eye_sores_or_infections: int
    Dry_mouth: int
    Headache_if_there_is_an_infection_such_as_Lyme_disease: int
    Loss_of_sense_of_taste: int
    Sound_that_is_louder_in_one_ear_hyperacusis: int
    Feeling_like_you_are_spinning_or_moving: int
    Feeling_like_the_world_is_spinning_around_you: int
    Vision_problems_such_as_a_feeling_that_things_are_jumping_or_moving: int
    Loss_of_feeling_sensation_in_hands_and_feet: int
    Loss_of_muscle_function_or_paralysis_of_the_lower_legs: int
    Strange_eye_movements_nystagmus: int
    Awakening_at_night_short_of_breath: int
    Urinary_incontinence: int
    Abdominal_pain_in_the_upper_right_side: int
    Eats_large_amounts_of_food_in_a_short_period: int
    Is_not_able_to_control_overeating: int
    Eats_food_very_fast_each_time: int
    Keeps_eating_even_when_full_gorging_or_until_uncomfortably_full: int
    Eats_even_though_not_hungry: int
    Eats_alone_in_secret: int
    Feels_guilty_disgusted_ashamed_or_depressed_after_eating_so_much: int
    Excess_involvement_in_activities: int
    Little_need_for_sleep: int
    Poor_judgment: int
    Poor_temper_control: int
    Lack_of_self_control_and_reckless_behavior: int
    Very_irritable_mood_racing_thoughts_talking_a_lot_and_false_beliefs_about_self_or_abilities: int
    Rapid_speech: int
    Concerns_about_things_that_are_not_true_delusions: int
    Painful_urination: int
    Continuous_feeling_of_a_full_bladder: int
    Pain_during_urination_dysuria: int
    Straining_to_urinate: int
    Urinary_tract_infection: int
    Waking_up_at_night_to_urinate_nocturia: int
    Inability_to_urinate_except_in_certain_positions: int
    Interruption_of_the_urine_stream: int
    Pain_discomfort_in_the_penis: int
    Signs_of_UTI_such_as_fever_pain_when_urinating_and_need_to_urinate_often: int
    Black_tarry_stools: int
    Symptoms_of_chronic_liver_disease: int
    Intense_fear_of_being_abandoned: int
    Cant_tolerate_being_alone: int
    Feelings_of_emptiness_and_boredom: int
    Displays_of_inappropriate_anger: int
    Impulsiveness_such_as_with_substance_use_or_sexual_relationships: int
    Self_injury_such_as_wrist_cutting_or_overdosing: int
    Weakness_with_paralysis: int
    Changes_in_mental_status_such_as_confusion_slow_response_or_thinking_unable_to_focus_or_sleepiness: int
    Decreased_ability_to_feel_touch_or_pain_sensation: int
    Loss_of_muscle_function_typically_on_one_side: int
    Weakness_in_a_limb: int
    Cardiac_arrest: int
    Wide_dilated_pupils_and_no_movement_in_one_or_both_eyes: int
    Small_pits_lumps_or_skin_tags_at_either_side_of_the_neck_or_just_below_the_jawbone: int
    Fluid_drainage_from_a_pit_on_the_neck: int
    Noisy_breathing: int
    Breast_lump_or_lump_in_the_armpit_that_is_hard_has_uneven_edges_and_usually_does_not_hurt: int
    Change_in_the_size_shape_or_feel_of_the_breast_or_nipple: int
    Fluid_from_the_nipple_fluid_may_be_bloody_clear_to_yellow_green_or_look_like_pus: int
    Muscles_around_the_ribs_sink_in_as_the_child_tries_to_breathe_in_called_intercostal_retractions: int
    Infants_nostrils_get_wide_when_breathing: int
    Reddish_or_purplish_color_in_a_circle_around_spider_bite: int
    Large_sore_ulcer_in_the_area_of_the_bite: int
    Red_thickened_skin_along_the_inside_edge_at_the_base_of_the_big_toe: int
    A_bony_bump_at_the_first_toe_joint_with_decreased_movement_in_the_toe_site: int
    Pain_over_the_joint_which_pressure_from_shoes_makes_worse: int
    Big_toe_turned_outward_toward_the_other_toes_and_may_cross_over_the_second_toe_as_a_result_corns_and_calluses_often_develop_where_the_first_and_second_toes_overlap: int
    Difficulty_wearing_regular_shoes: int
    Cramping_abdominal_pain: int
    Watery_diarrhea_sometimes_bloody: int
    Abnormal_posture_with_flexed_arms_and_straight_legs: int
    Feeding_problems: int
    Increasing_head_size: int
    Poor_muscle_tone_especially_of_the_neck_muscles: int
    A_lack_of_head_control_when_baby_is_pulled_from_a_lying_to_a_sitting_position: int
    Poor_visual_tracking_or_blindness: int
    Reflux_with_vomiting: int
    Severe_intellectual_disability: int
    One_or_more_painful_red_spots_or_bumps_that_develops_into_an_open_ulcer: int
    White_or_yellow_center_sores_on_the_inner_surface_of_the_cheeks_and_lips_tongue_upper_surface_of_the_mouth_and_the_base_of_the_gums: int
    Small_size_most_often_under_one_third_inch_or_1_centimeter_across_sores_on_the_inner_surface_of_the_cheeks_and_lips_tongue_upper_surface_of_the_mouth_and_the_base_of_the_gums: int
    Gray_color_as_healing_starts_on_the_sores_developed_on_the_inner_surface_of_the_cheeks_and_lips_tongue_upper_surface_of_the_mouth_and_the_base_of_the_gums: int
    Flushing_face_neck_or_upper_chest_or_widened_blood_vessels_seen_on_the_skin_telangiectasias: int
    Heart_problems_such_as_leaking_heart_valves_slow_heartbeat_low_or_high_blood_pressure: int
    Fatigue_reduced_exercise_ability: int
    Trouble_breathing_while_lying_down: int
    A_racing_heart: int
    Clumsiness_of_the_hand_when_gripping_objects: int
    Numbness_or_tingling_or_tingling_in_the_thumb_and_next_two_or_three_fingers_of_one_or_both_hands: int
    Numbness_or_tingling_of_the_palm_of_the_hand: int
    Pain_that_extends_to_the_elbow: int
    Pain_in_the_wrist_or_hand_in_one_or_both_hands: int
    Problems_with_fine_finger_movements_coordination_in_one_or_both_hands: int
    Wasting_away_of_the_muscle_under_the_thumb_in_advanced_or_long_term_cases: int
    Weak_grip: int
    Weakness_in_one_or_both_hands: int
    Being_sensitive_to_glare: int
    Cloudy_fuzzy_foggy_or_filmy_vision: int
    Difficulty_seeing_at_night_or_in_dim_light: int
    Double_vision: int
    Loss_of_color_intensity: int
    Problems_seeing_shapes_against_a_background_or_the_difference_between_shades_of_colors: int
    Seeing_halos_around_lights: int
    Frequent_changes_in_eyeglass_prescriptions: int
    Foul_or_strong_urine_odor: int
    Frequent_and_strong_urge_to_urinate: int
    Pressure_pain_or_spasms_in_your_back_or_the_lower_part_of_your_belly: int
    Bulging_eyeball_usually_on_one_side_of_face: int
    Cannot_move_the_eye_in_a_particular_direction: int
    Decreased_appetite_may_also_be_increased_or_unchanged: int
    Diarrhea_either_constant_or_off_and_on: int
    Lactose_intolerance_common_when_the_person_is_diagnosed_often_goes_away_after_treatment: int
    Stools_that_are_foul_smelling_oily_or_stick_to_the_toilet_when_flushed: int
    Fever_with_chills_and_sweating: int
    Pain_or_tenderness_in_the_affected_area: int
    Skin_redness_or_inflammation_that_gets_bigger_as_the_infection_spreads: int
    Skin_sore_or_rash_that_starts_suddenly_and_grows_quickly_in_the_first_24_hours: int
    Tight_glossy_stretched_appearance_of_the_skin: int
    Warm_skin_in_the_area_of_redness: int
    Muscle_aches_and_joint_stiffness_from_swelling_of_the_tissue_over_the_joint: int
    Dim_and_blurred_blind_spot_in_the_center_of_vision: int
    Distortion_of_straight_lines_with_the_affected_eye: int
    Objects_appearing_smaller_or_farther_away_with_the_affected_eye: int
    Nervous_system_changes_that_may_start_suddenly_including_confusion_delirium_double_vision_decreased_vision_sensation_changes_speech_problems_weakness_or_paralysis: int
    Episodes_of_confusion: int
    Headaches_that_come_and_go: int
    Loss_of_mental_function_dementia: int
    Weakness_or_unusual_sensations_that_come_and_go_and_involve_smaller_areas: int
    Ear_noise_buzzing_also_called_pulsatile_tinnitus: int
    Headache_in_one_or_more_parts_of_the_head_may_seem_like_a_migraine: int
    Muscle_weakness: int
    Numbness_in_an_area_of_the_body: int
    Change_in_attention_inattentiveness: int
    Speech_disorder: int
    Abnormal_vaginal_bleeding_between_periods_after_intercourse_or_after_menopause: int
    Vaginal_discharge_that_does_not_stop_and_may_be_pale_watery_pink_brown_bloody_or_foul_smelling: int
    Periods_that_become_heavier_and_last_longer_than_usual: int
    Vaginal_bleeding_after_douching_or_intercourse: int
    Abnormal_vaginal_bleeding_after_menopause_or_between_periods: int
    White_or_yellow_mucus_leukorrhea: int
    Neck_stiffness_that_gets_worse_over_time: int
    Numbness_or_abnormal_sensations_in_the_shoulders_or_arms: int
    Headaches_especially_in_the_back_of_the_head: int
    Pain_on_the_inside_of_the_shoulder_blade_and_shoulder_pain: int
    Unusual_vaginal_discharge_that_does_not_go_away_discharge_may_be_gray_white_or_yellow_in_color: int
    Painful_sexual_intercourse: int
    Pain_in_the_vagina: int
    Pressure_or_heaviness_in_the_pelvis: int
    Vaginal_itching: int
    Silver_hair_light_colored_eyes_albinism: int
    Increased_infections_in_the_lungs_skin_and_mucous_membranes: int
    Jerky_eye_movements_nystagmus: int
    The_blisters_are_most_often_first_seen_on_the_face_middle_of_the_body_or_scalp_after_a_day_or_two_the_blisters_become_cloudy_and_then_scab_meanwhile_new_blisters_form_in_groups_they_often_appear_in_the_mouth_in_the_vagina_and_on_the_eyelids: int
    Muscle_pain: int
    Discharge_from_the_penis_or_rectum: int
    Tenderness_or_pain_in_the_testicles: int
    Rectal_discharge_or_pain: int
    Symptoms_of_pelvic_inflammatory_disease_PID_salpingitis_inflammation_of_the_fallopian_tubes_or_liver_inflammation_similar_to_hepatitis: int
    Dark_urine: int
    Pain_in_the_upper_right_abdomen_that_may_radiate_to_the_back: int
    Pain_on_the_upper_right_side_or_upper_middle_part_of_the_abdomen_it_may_also_be_felt_in_the_back_or_below_the_right_shoulder_blade_the_pain_may_come_and_go_and_feel_sharp_cramp_like_or_dull: int
    Pain_in_the_right_upper_or_middle_upper_abdomen_for_at_least_30_minutes_the_pain_may_be_constant_and_intense_it_can_be_mild_or_severe: int
    Dry_skin: int
    Glassy_or_sunken_eyes: int
    Lack_of_tears: int
    Unusual_sleepiness_or_tiredness: int
    Inability_to_digest_certain_foods: int
    Pain_in_the_right_upper_part_of_the_abdomen: int
    Yellow_skin_or_eyes: int
    Sharp_cramping_or_dull_pain_in_upper_right_or_upper_middle_of_your_belly_lasting_for_30_minutes: int
    Problems_walking_due_to_weakness_or_lack_of_feeling_in_the_feet: int
    Trouble_using_the_arms_and_hands_or_legs_and_feet_due_to_weakness: int
    Sensation_changes_such_as_numbness_or_decreased_sensation_pain_burning_tingling_or_other_abnormal_sensations_usually_affects_the_feet_first_then_the_arms_and_hands: int
    Enlarged_lymph_nodes_liver_or_spleen: int
    Infections_that_keep_coming_back_recur_despite_treatment: int
    Excessive_blinking: int
    Grimaces_of_the_face: int
    Quick_movements_of_the_arms_legs_or_other_areas: int
    Sounds_grunts_throat_clearing_contractions_of_the_abdomen_or_diaphragm: int
    Pressure_under_the_lower_left_ribs_from_a_swollen_spleen: int
    Cough_with_or_without_mucus: int
    Frequent_respiratory_infections: int
    Trouble_catching_ones_breath: int
    Abdominal_pain_may_last_from_hours_to_days_over_time_may_always_be_present: int
    Abdominal_pain_may_get_worse_from_eating: int
    Abdominal_pain_may_get_worse_from_drinking_alcohol: int
    Chronic_weight_loss_even_when_eating_habits_and_amounts_are_normal: int
    Decreased_memory: int
    Problem_speaking: int
    Weakness_or_numbness_of_arms_legs_face: int
    Enlarged_neck_or_presence_of_goiter_which_may_be_the_only_early_symptom: int
    Hair_loss: int
    Intolerance_to_cold: int
    Weight_gain: int
    Small_or_shrunken_thyroid_gland_late_in_the_disease: int
    Abdominal_pain_due_to_pancreatitis_inflammation_of_the_pancreas: int
    Symptoms_of_nerve_damage_such_as_loss_of_feeling_in_the_feet_or_legs_and_memory_loss: int
    Yellow_deposits_of_fatty_material_in_the_skin_called_xanthomas_these_growthsmay_appear_on_the_back_buttocks_soles_of_the_feet_or_ankles_knees_and_elbows: int
    Small_red_spider_like_blood_vessels_on_the_skin: int
    Fluid_buildup_in_the_legs_edema_and_in_the_abdomen_ascites: int
    Redness_on_the_palms_of_the_hands: int
    In_men_impotence_shrinking_of_the_testicles_and_breast_swelling: int
    Easy_bruising_and_abnormal_bleeding_most_often_from_swollen_veins_in_the_digestive_tract: int
    Bleeding_from_upper_or_lower_gastrointestinal_tract: int
    Ability_to_touch_shoulders_together_in_front_of_body: int
    Delayed_closure_of_fontanelles_soft_spots: int
    Loose_joints: int
    Prominent_forehead_frontal_bossing: int
    Short_forearms: int
    Short_fingers: int
    Short_stature: int
    Increased_risk_of_getting_flat_foot_abnormal_curvature_of_spine_scoliosis_and_knee_deformities: int
    High_risk_of_hearing_loss_due_to_infections: int
    Increased_risk_of_fracture_due_to_decreased_bone_density: int
    Blind_spots: int
    Floaters: int
    Cold_feet_or_legs: int
    Failure_to_thrive: int
    Constant_urge_to_have_a_bowel_movement: int
    Sharp_chest_or_shoulder_pain_made_worse_by_a_deep_breath_or_a_cough: int
    Nasal_flaring: int
    Feeling_weak_all_over_and_muscle_aches: int
    Headache_behind_the_eyes_typically_during_fever: int
    Skin_pain: int
    Abdominal_pain_and_tenderness_in_the_lower_abdomen: int
    Narrow_stools: int
    Scratchy_throat: int
    Decreased_sensation_numbness_or_tingling_in_the_top_of_the_foot_or_the_outer_part_of_the_upper_or_lower_leg: int
    Foot_that_drops_unable_to_hold_the_foot_up: int
    Slapping_gait_walking_pattern_in_which_each_step_makes_a_slapping_noise: int
    Toes_drag_while_walking: int
    Weakness_of_theankles_or_feet: int
    Loss_of_muscle_mass_because_the_nerves_arent_stimulating_the_muscles: int
    Pain_that_is_much_higher_than_expected_with_the_injury: int
    Severe_pain_that_doesnt_go_away_after_taking_pain_medicine_or_raising_the_affected_area: int
    Decreased_sensation_numbness_tingling_weakness_of_the_affected_area: int
    Changes_in_skin_temperature_switching_between_warm_or_cold: int
    Faster_growth_of_nails_and_hair: int
    Severe_burning_aching_pain_that_worsens_with_the_slightest_touch_or_breeze: int
    Skin_that_slowly_becomes_blotchy_purple_pale_or_red_thin_and_shiny_swollen_more_sweaty: int
    Acting_somewhat_confused_feeling_unable_to_concentrate_or_not_thinking_clearly: int
    Being_drowsy_hard_to_wake_up_or_similar_changes: int
    Memory_loss_amnesia_of_events_before_the_injury_or_right_after: int
    Seeing_flashing_lights_light_sensitivity: int
    Feeling_like_you_have_lost_time: int
    Sleep_abnormalities: int
    Breaking_rules_without_clear_reason: int
    Cruel_or_aggressive_behavior_toward_people_or_animals_for_example_bullying_fighting_using_dangerous_weapons_forcing_sexual_activity_and_stealing: int
    Not_going_to_school_truancy_beginning_before_age_13: int
    Heavy_drinking_or_drug_use: int
    Intentionally_setting_fires: int
    Lying_to_get_a_favor_or_avoid_things_they_have_to_do: int
    Running_away: int
    Vandalizing_or_destroying_property: int
    An_infant_does_not_seem_to_be_visually_aware_of_the_world_around_them_if_cataracts_are_in_both_eyes: int
    Gray_or_white_cloudiness_of_the_pupil_which_is_normally_black: int
    The_red_eye_glow_red_reflex_of_the_pupil_is_missing_in_photos_or_is_different_between_the_2_eyes: int
    Unusual_rapid_eye_movements_nystagmus: int
    Inflammation_of_the_retina: int
    Low_birth_weight: int
    Mineral_deposits_in_the_brain: int
    Bleeding_from_the_umbilical_cord_just_after_birth: int
    Bleeding_in_the_mucous_membranes: int
    Bleeding_in_the_brain: int
    Bleeding_in_the_joints: int
    Heavy_bleeding_after_injury_or_surgery: int
    Nosebleeds_that_do_not_stop_easily: int
    Foamy_appearance_of_urine: int
    Crusts_that_form_on_the_eyelid_overnight_most_often_caused_by_bacteria: int
    Eye_pain: int
    Gritty_feeling_in_the_eyes: int
    Body_aches: int
    Enlarged_pupil_that_does_not_get_smaller_when_a_light_shines_on_it:int
    Increasing_pressure_on_the_brain_usually_from_hydrocephalus:int
    Disrupting_hormone_production_by_the_pituitary_gland:int
    Pressure_or_damage_to_the_optic_nerve:int
    Dementia_that_gets_worse_quickly_over_a_few_weeks_or_months:int
    Changes_in_gait_walking_:int
    Feeling_nervous:int
    Feeling_that_you_need_to_pass_stools_even_though_your_bowels_are_already_empty_tenesmus_it_may_involve_straining_pain_and_cramping:int
    Watery_diarrhea_which_may_be_blood:int
    Glomerulonephritis_kidney_inflammation_:int
    Purpura:int
    Skin_death:int
    Skin_ulcers:int
    Diarrhea_which_is_often_watery_non_bloody_large_volume_and_occurs_many_times_a_day:int
    A_bend_in_the_penis_which_most_often_begins_at_the_area_where_you_feel_the_scar_tissue_or_hardening:int
    Softening_of_the_portion_of_the_penis_beyond_the_area_of_scar_tissue:int
    Narrowing_of_the_penis:int
    Problems_with_penetration_or_pain_during_intercourse:int
    Shortening_of_the_penis:int
    Pain_or_burning_with_urination:int
    Pressure_or_cramping_in_the_lower_middle_abdomen_or_back:int
    Strong_need_to_urinate_often_even_right_after_the_bladder_has_been_emptied:int
    Enlarged_lymph_nodes_especially_in_the_neck:int
    Skin_that_feels_warm_to_the_touch:int
    Dry_or_sticky_mouth:int
    Changes_in_alertness_usually_more_alert_in_the_morning_less_alert_at_night_:int
    Changes_in_feeling_sensation_and_perception:int
    Decrease_in_short_term_memory_and_recall:int
    Disorganized_thinking_such_as_talking_in_a_way_that_doesn_t_make_sense:int
    Emotional_or_personality_changes_such_as_anger_agitation_depression_irritability_and_overly_happy:int
    Incontinence:int
    Movements_triggered_by_changes_in_the_nervous_system:int
    Problem_concentrating:int
    Changes_in_mental_function:int
    Deep_sleep_that_lasts_for_a_day_or_longer:int
    Sudden_severe_confusion_delirium:int
    Excitement_or_fear:int
    Bursts_of_energy:int
    Quick_mood_changes:int
    Sensitivity_to_light_sound_touch:int
    Difficulty_doing_more_than_one_task_at_a_time:int
    Difficulty_solving_problems_or_making_decisions:int
    Forgetting_names_of_familiar_people_recent_events_or_conversations:int
    Taking_longer_to_do_more_difficult_mental_activities:int
    Personality_changes_and_loss_of_social_skills_which_can_lead_to_inappropriate_behaviors:int
    Losing_interest_in_things_previously_enjoyed_flat_mood:int
    Getting_lost_on_familiar_routes_1:int
    Avoiding_being_alone:int
    Avoiding_personal_responsibility:int
    Becoming_easily_hurt_by_criticism_or_disapproval:int
    Becoming_overly_focused_on_fears_of_being_abandoned:int
    Becoming_very_passive_in_relationships:int
    Feeling_very_upset_or_helpless_when_relationships_end:int
    Having_difficulty_making_decisions_without_support_from_others:int
    Having_problems_expressing_disagreements_with_others:int
    Extremely_itchy_bumps_or_blisters_most_often_on_the_elbows_knees_back_and_buttocks:int
    Rashes_that_are_usually_the_same_size_and_shape_on_both_sides:int
    The_rash_can_look_like_eczema:int
    Scratch_marks_and_skin_erosions_instead_of_blisters_in_some_people:int
    Purple_color_to_the_upper_eyelids:int
    Purple_red_skin_rash:int
    Clumsiness:int
    Delays_in_sitting_up_crawling_and_walking:int
    Problems_with_sucking_and_swallowing_during_first_year_of_life:int
    Problems_with_gross_motor_coordination_for_example_jumping_hopping_or_standing_on_one_foot:int
    Problems_with_visual_or_fine_motor_coordination_for_example_writing_using_scissors_tying_shoelaces_or_tapping_one_finger_to_another:int
    Hunger:int
    Chest_pain_or_pressure_more_likely_with_exercise:int
    swelling_of_ankles:int
    Croup_like_barking_cough:int
    Bloody_watery_drainage_from_nose:int
    Difficulty_getting_up_and_standing:int
    Increased_curvature_of_the_back:int
    Sweating_at_night:int
    Recent_flu_like_symptoms:int
    Refusal_to_sit_up_stand_or_walk_younger_child:int
    neck_pain:int
    back_stiffness:int
    Bleeding_from_many_sites_in_the_body:int
    Blood_clots:int
    Drop_in_blood_pressure:int
    Pale_skin_due_to_anemia_pallor:int
    Swollen_glands:int
    Impaired_growth_in_children:int
    Nephrocalcinosis_too_much_calcium_deposited_in_the_kidneys:int
    Osteomalacia_softening_of_the_bones:int
    Tenderness_usually_in_the_left_lower_part_of_the_abdomen:int
    Not_feeling_hungry_and_not_eating:int
    High_pitched_sound_during_breathing_stridor:int
    Repeated_pneumonias:int
    Decreased_muscle_tone_at_birth:int
    Excess_skin_at_the_nape_of_the_neck:int
    Flattened_nose:int
    Separated_joints_between_the_bones_of_the_skull_sutures:int
    Single_crease_in_the_palm_of_the_hand_1:int
    Small_ears:int
    Small_mouth:int
    Upward_slanting_eyes:int
    Wide_short_hands_with_short_fingers:int
    White_spots_on_the_colored_part_of_the_eye_Brushfield_spots:int
    Sounding_as_though_they_are_mumbling:int
    Speaking_softly_or_in_a_whisper:int
    Speaking_in_a_nasal_or_stuffy_hoarse_strained_or_breathy_voice:int
    Errors_in_grammar_and_punctuation:int
    Poor_handwriting:int
    Poor_spelling:int
    Poorly_organized_writing:int
    The_need_to_say_words_aloud_when_writing:int
    Ear_discomfort_or_pain_in_one_or_both_ears:int
    Sensation_of_fullness_or_stuffiness_in_the_ears:int
    Drainage_of_fluid_from_the_ear:int
    Pain_in_the_upper_right_part_of_the_abdomen:int
    Severe_allergic_reaction:int
    Mouth_sores:int
    Abnormal_nails:int
    Abnormal_or_missing_teeth_or_fewer_than_normal_number_of_teeth:int
    Decreased_skin_color_pigment:int
    Low_nasal_bridge:int
    Thin_sparse_hair:int
    Learning_disabilities:int
    Poor_hearing:int
    Poor_vision_with_decreased_tear_production:int
    Weakened_immune_system:int
    Double_jointedness:int
    Easy_scarring_and_poor_wound_healing:int
    Flat_feet:int
    Increased_joint_mobility_joints_popping_early_arthritis:int
    Joint_dislocation:int
    Very_soft_and_velvety_skin:int
    Rounded_fingernails_and_toenails_clubbing:int
    Numbness_and_tingling_of_fingers_and_toes:int
    Swelling_in_the_joints_caused_by_too_much_uric_acid_gout:int
    Epispadias_or_undescended_testicle_cryptorchidism:int
    Extra_fingers_polydactyly:int
    Limited_range_of_motion:int
    Nail_problems_including_missing_or_deformed_nails:int
    Short_arms_and_legs_especially_forearm_and_lower_leg:int
    Short_height_between_3_5_to_5_feet_1_to_1_5_meters_tall:int
    Sparse_absent_or_fine_textured_hair:int
    Tooth_abnormalities_such_as_peg_teeth_widely_spaced_teeth:int
    Teeth_present_at_birth_natal_teeth:int
    Chest_pain_which_worsens_when_you_breathe_in_deeply_pleurisy:int
    Dry_cough:int
    Menstrual_bleeding_that_is_not_regular_or_predictable:int
    Trouble_getting_or_staying_pregnant_infertility:int
    Painful_periods_Cramps_or_pain_in_your_lower_belly_may_begin_a_week_or_two_before_your_period_cramps_may_be_steady_and_range_from_dull_to_severe:int
    Pain_with_bowel_movements:int
    Long_term_pelvic_or_low_back_pain_that_may_occur_at_any_time_and_last_for_6_months_or_more:int
    Bad_breath:int
    Cracked_lips:int
    Dribbling_at_the_end_of_urinating:int
    Incomplete_emptying_of_your_bladder:int
    Needing_to_urinate_2_or_more_times_per_night:int
    Slowed_or_delayed_start_of_the_urinary_stream:int
    Weak_urine_stream:int
    Tenderness_and_swelling_of_the_skin_on_the_arms_legs_or_sometimes_the_joints_most_often_on_both_sides_of_the_body:int
    Arthritis:int
    Thickened_skin_that_looks_puckered:int
    Tender_or_sore_skin:int
    Warm_skin_in_the_affected_area:int
    Blisters_around_the_eyes_and_nose:int
    Blisters_in_or_around_the_mouth_and_throat_causing_feeding_problems_or_swallowing_difficulty:int
    Blisters_on_the_skin_as_a_result_of_minor_injury_or_temperature_change_especially_of_the_feet:int
    Blistering_that_is_present_at_birth:int
    Hoarse_cry_cough_or_other_breathing_problems:int
    Tiny_white_bumps_on_previously_injured_skin:int
    Mitten_like_hands_or_feet:int
    Thickening_of_skin_on_hands_and_feet:int
    Bowel_or_bladder_incontinence:int
    Decreased_ability_to_move_any_part_of_the_body:int
    Enlarged_pupil_in_one_eye:int
    Abnormal_breathing_sounds_stridor:int
    Violent_shaking_and_loss_of_alertness:int
    Staring_spells:int
    Sensations_may_be_tingling_smelling_an_odor_that_is_not_actually_there_or_emotional_changes:int
    A_pink_or_purple_color_to_the_normally_white_part_of_the_eye:int
    Eye_tenderness:int
    Tearing_of_the_eye:int
    Waxy_skin_bumps:int
    Firm_skin_bumps:int
    Throbbing_pain_with_itching_or_burning_sensation:int
    Fluid_filled_blisters:int
    Redness_over_80_to_90_of_the_body:int
    Scaly_skin_patches:int
    Loss_of_temperature_regulation_by_the_body:int
    Backward_movement_of_food_through_the_esophagus_and_possibly_mouth_regurgitation:int
    Headache_between_the_eyes:int
    Eye_blinking:int
    Mouth_twitching:int
    Nose_wrinkling:int
    Squinting:int
    Inability_to_whistle_due_to_weakness_of_the_cheek_muscles:int
    Decreased_facial_expression_due_to_weakness_of_facial_muscles:int
    Depressed_or_angry_facial_expression:int
    Difficulty_pronouncing_words:int
    Difficulty_reaching_above_the_shoulder_level:int
    Swallowing_problems_in_infants_resulting_in_aspiration_pneumonia_or_poor_growth:int
    Breath_holding_spells_resulting_in_fainting:int
    Inability_to_feel_pain_and_changes_in_temperature_can_lead_to_injuries:int
    Unusually_smooth_pale_tongue_surface_and_lack_of_taste_buds_and_decrease_in_sense_of_taste:int
    Fatty_skin_deposits_called_xanthomas_over_parts_of_the_hands_elbows_knees_ankles_and_around_the_cornea_of_the_eye:int
    Cholesterol_deposits_in_the_eyelids_xanthelasmas:int
    Chest_pain_angina_or_other_signs_of_coronary_artery_disease_may_be_present_at_a_young_age:int
    Cramping_of_one_or_both_calves_when_walking:int
    Sudden_stroke_like_symptoms_such_as_trouble_speaking_drooping_on_one_side_of_the_face_weakness_of_an_arm_or_leg_and_loss_of_balance:int
    High_triglyceride_levels_in_the_blood:int
    Chronic_inflammation_of_the_pancreas:int
    Skin_sores_lesions_that_are_red_and_swollen_and_range_from_5_to_20_cm_in_diameter:int
    Eye_strain:int
    Headache_while_reading:int
    Leakage_of_liquid_or_sudden_episodes_of_watery_diarrhea_in_someone_who_has_chronic_long_term_constipation:int
    Rectal_bleeding:int
    Small_semi_formed_stools:int
    Straining_when_trying_to_pass_stools:int
    Recurrent_infections:int
    Sudden_groin_or_thigh_pain:int
    Pain_or_discomfort_in_both_breasts_that_may_come_and_go_with_your_period_but_may_last_through_the_whole_month:int
    Breasts_that_feel_full_swollen_or_heavy:int
    Pain_or_discomfort_under_the_arms:int
    Breast_lumps_that_change_in_size_with_the_menstrual_period:int
    Bone_sores_lesions:int
    Hormone_endocrine_gland_problems:int
    Unusual_skin_color_pigmentation_which_occurs_with_McCune_Albright_syndrome:int
    Flushed_face:int
    Swelling_called_generalized_edema_from_fluids_held_in_the_body:int
    Abnormal_sensations_such_as_numbness_tingling_crawling_sensation_like_ants_crawling_on_the_skin:int
    Dilated_pupils:int
    Autism_spectrum_disorder:int
    Hand_flapping_or_hand_biting:int
    Speech_and_language_delay:int
    Tendency_to_avoid_eye_contact:int
    Changes_in_vision_particularly_color_vision:int
    Decrease_in_ability_to_feel_vibrations_in_lower_limbs:int
    Foot_problems_such_as_hammer_toe_and_high_arches:int
    No_reflexes_in_the_legs:int
    Seeing_colored_halos_around_lights:int
    Worsening_vision_throughout_the_day:int
    Pain_in_the_right_upper_or_middle_upper_abdomen_for_at_least_30_minutes_the_pain_may_be_constant_or_cramping_it_can_feel_sharp_or_dull:int
    Discoloration_blue_or_black_if_skin_is_affected_red_or_bronze_if_the_affected_area_is_beneath_the_skin:int
    Loss_of_feeling_in_the_area_which_may_happen_after_severe_pain_in_the_area:int
    Air_under_the_skin:int
    Drainage_from_the_tissues_foul_smelling_brown_red_or_bloody_fluid:int
    Moderate_to_severe_pain_around_a_skin_injury:int
    Pale_skin_color_later_becoming_dusky_and_changing_to_dark_red_or_purple:int
    Swelling_that_worsens_around_a_skin_injury:int
    Vesicle_formation_combining_into_large_blisters:int
    Hypoglycemia:int
    Premature_abdominal_fullness_after_meals:int
    Problems_falling_or_staying_asleep_or_sleep_that_is_restless_and_unsatisfying:int
    Restlessness_when_awake:int
    Muscle_aches_in_the_lower_back_buttocks_thighs_or_knees:int
    Rash_or_patch_on_the_skin_usually_on_the_arms_and_legs:int
    Brownish_red_or_copper_colored_patch_that_is_firm_and_flat_on_top:int
    String_of_bumps_may_appear_in_a_line:int
    Rash_may_appear_on_the_palms_and_soles_but_not_on_the_back_chest_or_belly_area:int
    Bright_red_or_reddish_purple_gums:int
    Gums_that_are_tender_when_touched_but_otherwise_painless:int
    Swollen_gums:int
    Shiny_appearance_to_gums:int
    Sores_on_the_inside_of_the_cheeks_or_gums:int
    Very_sore_mouth_with_no_desire_to_eat:int
    Rainbow_like_halos_around_lights:int
    Weakness_or_loss_of_movement_in_the_face:int
    Smooth_surface_of_the_tongue:int
    Pale_or_bright_red_color_to_the_tongue:int
    pain_in_back_of_the_nose_and_throat_nasopharynx:int
    pain_in_back_of_the_tongue:int
    pain_in_tonsil_area:int
    Glucose_intolerance_body_has_problem_breaking_down_sugars:int
    High_blood_sugar:int
    Increased_appetite:int
    Pain_in_the_hands_or_wrists_due_to_tendon_inflammation:int
    Red_or_swollen_opening_of_penis_urethra:int
    Abnormal_uterine_bleeding:int
    Bleeding_after_sex:int
    Abnormal_vaginal_discharge_with_greenish_yellow_or_foul_smelling_discharge:int
    joint_appears_warm_and_red:int
    Mental_status_changes:int
    Possible_skin_thickening:int
    Mild_itching_of_the_skin:int
    Daytime_naps_that_do_not_relieve_drowsiness:int
    Difficulty_waking_from_a_long_sleep_may_feel_confused_or_disoriented_sleep_drunkenness:int
    Increased_need_for_sleep_during_the_day_even_while_at_work_or_during_a_meal_or_conversation:int
    Increased_sleep_time_up_to_14_to_18_hours_a_day:int
    Headaches_throbbing_daily_irregular_and_worse_in_the_morning:int
    Low_back_pain_radiating_along_both_legs:int
    Not_able_to_be_as_active_as_before:int
    Pain_or_fullness_in_the_upper_left_belly_enlarged_spleen:int
    A_tight_band_around_the_chest:int
    Squeezing_or_heavy_pressure_on_chest:int
    Something_heavy_sitting_on_your_chest:int
    Loss_of_body_hair:int
    Gastrointestinal_tract_and_urinary_tract_bleeding:int
    Bleeding_that_starts_without_cause:int
    Painless_bright_red_blood_from_the_rectum:int
    Pain_in_the_right_upper_abdomen:int
    unable_to_pass_stools:int
    sharp_pain_in_one_part_of_the_leg_hip_or_buttocks_and_numbness_in_other_parts:int
    pain_or_numbness_on_the_back_of_the_calf_or_sole_of_the_foot:int
    pain_when_moving_your_neck_deep_pain_near_or_over_the_shoulder_blade_or_pain_that_moves_to_the_upper_arm_forearm_and_fingers:int
    numbness_along_your_shoulder_elbow_forearm_and_fingers:int
    pain_get_worse_after_standing_or_sitting:int
    pain_get_worse_at_night:int
    pain_get_worse_when_sneezing_coughing_or_laughing:int
    pain_gets_worse_when_bending_backward_or_walking_more_than_a_few_yards_or_meters:int
    pain_get_worse_when_straining_or_holding_your_breath_such_as_when_having_a_bowel_movement:int
    Ulcers_in_the_mouth_and_throat_and_similar_sores_on_the_feet_hands_and_buttocks:int
    Itching_of_the_lips_or_skin_around_the_mouth:int
    Burning_near_the_lips_or_mouth_area:int
    Tingling_near_the_lips_or_mouth_area:int
    Blisters_in_the_mouth_often_on_the_tongue_cheeks_roof_of_the_mouth_gums_and_on_the_border_between_the_inside_of_the_lip_and_the_skin_next_to_it:int
    After_blisters_pop_they_form_ulcers_in_the_mouth_often_on_the_tongue_or_cheeks:int
    Mouth_pain:int
    Shortened_foot_length:int
    Difficulty_fitting_shoes:int
    Foot_pain_with_walking_standing_and_running:int
    Sudden_collapse_when_the_heartbeat_gets_too_slow_or_even_stops:int
    Red_skin_bumps_most_often_on_the_lower_legs:int
    Acting_or_looking_overly_seductive:int
    Being_easily_influenced_by_other_people:int
    Being_overly_concerned_with_their_looks:int
    Being_overly_dramatic_and_emotional:int
    Being_overly_sensitive_to_criticism_or_disapproval:int
    Believing_that_relationships_are_more_intimate_than_they_actually_are:int
    Blaming_failure_or_disappointment_on_others:int
    Constantly_seeking_reassurance_or_approval:int
    Having_a_low_tolerance_for_frustration_or_delayed_gratification: int
    Needing_to_be_the_center_of_attention_self_centeredness:int
    Quickly_changing_emotions_which_may_seem_shallow_to_others:int
    Intestinal_infection:int
    Swelling_of_the_surface_of_the_skin_into_red_or_skin_colored_welts_called_wheals_with_clearly_defined_edges:int
    Wheals_may_get_bigger_spread_and_join_together_to_form_larger_areas_of_flat_raised_skin:int
    Wheals_often_change_shape_disappear_and_reappear_within_minutes_or_hours_it_is_unusual_for_a_wheal_to_last_more_than_48_hours:int
    Feeling_very_tired_all_the_time:int
    Fever_and_chills_that_come_and_go:int
    Itching_all_over_the_body_that_cannot_be_explained:int
    Drenching_night_sweats:int
    Chest_deformities:int
    Flush_across_the_cheeks:int
    High_arches_of_the_feet:int
    Knock_knees:int
    Long_limbs:int
    Mental_disorders:int
    Nearsightedness:int
    Spidery_fingers:int
    Abdominal_discomfort:int
    Decreased_sweating_on_the_affected_side_of_the_face:int
    Sinking_of_the_eyeball_into_the_face:int
    Different_sizes_of_pupils_of_the_eyes_anisocoria_with_the_affected_side_pupil_being_smaller:int
    Vertigo_sensation_that_surroundings_are_spinning_with_nausea_and_vomiting:int
    One_sided_neck_and_ear_pain:int
    Overreaction_of_the_involuntary_autonomic_nervous_system_to_stimulation_hyperreflexia:int
    Behavioral_disturbances:int
    Moodiness:int
    Paranoia:int
    Psychosis:int
    Fragile_bones_of_the_limbs_and_spine_that_can_break_easily:int
    New_rash_with_tender_purple_or_brownish_red_spots_over_large_areas:int
    Skin_sores_mostly_located_on_the_legs_buttocks_or_trunk:int
    Blisters_on_the_skin:int
    Hives_urticaria_may_last_longer_than_24_hours:int
    Open_sores_with_dead_tissue_necrotic_ulcers:int
    Low_level_of_one_or_more_types_of_blood_cells:int
    Stomach_pain_on_the_left_side:int
    Heat_intolerance:int
    Nervousness:int
    Bleeding_in_the_anterior_chamber_of_the_eye:int
    Loss_of_interest_in_sex:int
    Loss_of_menstrual_periods:int
    Infertility:int
    n_boys_no_development_of_sex_characteristics_such_as_enlargement_of_the_testes_and_penis_deepening_of_the_voice_and_facial_hair:int
    In_girls_a_lack_of_breast_development_and_menstrual_periods:int
    Bleeding_into_the_skin_often_around_the_shins_causing_a_skin_rash_that_looks_like_pinpoint_red_spots_petechial_rash:int
    One_or_many_blisters_that_are_filled_with_pus_and_easy_to_pop_in_infants_the_skin_is_reddish_or_raw_looking_where_a_blister_has_broken:int
    Blisters_that_itch_are_filled_with_yellow_or_honey_colored_fluid_and_ooze_and_crust_over_rash_that_may_begin_as_a_single_spot_but_spreads_to_other_areas_due_to_scratching:int
    Skin_sores_on_the_face_lips_arms_or_legs_that_spread_to_other_areas:int
    Swollen_lymph_nodes_near_the_infection:int
    Trouble_falling_asleep_on_most_nights:int
    Feeling_tired_during_the_day_or_falling_asleep_during_the_day:int
    Not_feeling_refreshed_when_you_wake_up:int
    Waking_up_several_times_during_sleep:int
    Yellow_color_inside_the_mouth:int
    Mild_to_severe_pain:int
    Inflammation_of_ear_canal:int
    Small_bumps_that_look_like_goose_bumps_on_the_back_of_the_upper_arms_and_thighs:int
    Bumps_feel_like_very_rough_sandpaper:int
    Skin_colored_bumps_are_the_size_of_a_grain_of_sand:int
    Slight_pinkness_may_be_seen_around_some_bumps:int
    Bumps_may_appear_on_the_face_and_be_mistaken_for_acne:int
    Pain_may_be_felt_in_the_belly_area_or_side_of_the_back:int
    Pain_may_move_to_the_groin_area_groin_pain_testicles_testicle_pain_in_men_and_labia_vaginal_pain_in_women:int
    Urethra_on_the_underside_of_the_penis:int
    Testicles_that_have_not_moved_into_the_scrotum:int
    Being_slow_to_sit_up_walk_crawl_and_speak:int
    Delayed_puberty:int
    Enlarged_breasts:int
    Small_penis_size:int
    Sexual_problems:int
    Changes_in_skin_pigment:int
    Increased_and_more_severe_infections_due_to_damaged_immune_system:int
    Hair_changes_change_in_color_or_texture:int
    Large_belly_that_sticks_out_protrudes:int
    Your_eyes_moving_on_their_own_making_it_hard_to_focus_them:int
    stuffy_nose:int
    Skin_lesions_that_are_lighter_than_your_normal_skin_color:int
    Lesions_that_have_decreased_sensation_to_touch_heat_or_pain:int
    Lesions_that_do_not_heal_after_several_weeks_to_months:int
    Enlargement_of_a_testicle_or_change_in_the_way_it_feels:int
    Excess_growth_of_breast_tissue:int
    Heaviness_in_the_scrotum:int
    Lump_or_swelling_in_either_testicle:int
    Abnormal_sometimes_waddling_walk:int
    Joints_that_are_fixed_in_a_contracted_position:int
    Large_and_muscular_looking_calves_pseudohypertrophy_which_are_not_actually_strong: int
    Loss_of_muscle_mass_thinning_of_certain_body_parts:int
    Tongue_swelling_or_protrusion_of_the_tongue_out_of_the_mouth:int
    tooth_pain:int
    Neck_swelling:int
    Redness_of_the_neck:int
    Bulky_stools:int
    Anemia:int
    muscle_rigidity:int
    Rise_in_body_temperature_to_105F_40_6C_or_higher:int
    Ongoing_drainage_from_the_ear_that_is_yellow_or_green_and_smells_bad:int
    Ear_pain_deep_inside_the_ear_pain_may_get_worse_when_you_move_your_head:int
    Itching_of_the_ear_or_ear_canal:int
    Limited_ability_to_tolerate_exercise:int
    Urine_that_smells_like_maple_syrup:int
    A_chest_that_sinks_in_or_sticks_out_called_funnel_chest_pectus_excavatum_or_pigeon_breast_pectus_carinatum:int
    Highly_arched_palate_and_crowded_teeth:int
    Joints_that_are_too_flexible_but_the_elbows_may_be_less_flexible:int
    Movement_of_the_lens_of_the_eye_from_its_normal_position_dislocation:int
    Spine_that_curves_to_one_side:int
    Thin_narrow_face:int
    Drainage_from_the_ear:int
    Redness_of_the_ear_or_behind_the_ear:int
    Swelling_behind_the_ear_may_cause_ear_to_stick_out_or_feel_as_if_it_is_filled_with_fluid:int
    Bloodshot_eyes:int
    Tiny_white_spots_inside_the_mouth_Koplik_spots:int
    Abnormal_strength_and_direction_of_urine_stream:int
    Bed_wetting:int
    Bleeding_hematuria_at_end_of_urination:int
    Visible_narrowing_of_the_urethral_opening_in_boys:int
    Swelling_and_redness_of_eyelid_edges:int
    Slight_blurring_of_vision_due_to_excess_oil_in_tears:int
    Frequent_styes_bumps:int
    Pressure_in_the_ear:int


        # class Config:
        #     arbitrary_types_allowed = True


dbfile = open('symptoms_dictionary', 'rb')    
db = pickle.load(dbfile)
model = pickle.load(open('model.sav', 'rb'))
# model = pickle.load(open('model_lr.sav', 'rb'))

@app.post('/dp_pred')
def dp_predictor(input_parameters:model_input):
    input_data = input_parameters.model_dump_json()
    input_dictionary = json.loads(input_data)
    input_list = list(input_dictionary.values())
    print("input values are = ", input_list)

    result=model.predict_proba([input_list])
    result=result.reshape(-1)
    c=0
    res={}
    for i in db:
        res[result[c]]=i
        c+=1
    
    
    myKeys = list(res.keys())
    myKeys.sort(reverse=True)
    sorted_dict = {i: res[i] for i in myKeys}
    # res = sorted(res.items(), reverse=True)
    # res=dict(res)
    # len(res)
    # c=1
    keys=[5]
    # values=[5]
    # for i in res:
    #     value={j for j in db if db[j]==res[i]}
    #     print(f'{value} - {i*100}%')
    #     keys.append(f'{value} - {i*100}%')
    #     c+=1
    #     if(c>5):break

    c=1
    for i in sorted_dict:
        print(f'{sorted_dict[i]}')
        keys.append(f'{sorted_dict[i]}')
        c+=1
        if c>5:
            break

    send=json.dumps(keys)
    print("Sending:",send)
    return send