from config import PALM_API_KEY
import time
import os
import csv
from collections import deque
import pandas as pd
import numpy as np
import re
#from sklearn.model_selection import train_test_split
from datasets import Dataset
import google.generativeai as palm
import google.ai.generativelanguage as safety_types

palm.configure(api_key=palm_api_key)
models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
model = models[0].name #models/text-bison-001
print(model)

df = pd.read_csv("/Users/christopherackerman/input/train_cln.csv")
df['keyword'] = df['keyword'].fillna('unknown')
df['location'] = df['location'].fillna('unknown')
sep = " [SEP] "
df['text'] = df.apply(lambda row: f"{row['text']} {sep} keyword: {row['keyword']} {sep} location: {row['location']}", axis=1)
train_ds = Dataset.from_pandas(df)
train_ds = train_ds.train_test_split(test_size=0.05, seed=42)

df = pd.read_csv("/Users/christopherackerman/input/test.csv")
df['keyword'] = df['keyword'].fillna('unknown')
df['location'] = df['location'].fillna('unknown')
sep = " [SEP] "
df['text'] = df.apply(lambda row: f"{row['text']} {sep} keyword: {row['keyword']} {sep} location: {row['location']}", axis=1)
test_ds = Dataset.from_pandas(df)

#from sklearn.model_selection import KFold
#k = 4  # 25% val set is okay
#kf = KFold(n_splits=k, shuffle=True, random_state=42)
#for k_i, (train_index, val_index) in enumerate(kf.split(ds)):
#    train_ds = ds.select(train_index)
#    val_ds = ds.select(val_index)
for dset in ["eval_short", "test"]:
    if "eval" in dset:
        ds = train_ds['test']
    else:
        ds = test_ds
    # Initialize the count of correct predictions
    correct_predictions = 0
    true_label=''

    # Initialize a list to store the text responses
    responses = []

    queue = deque(maxlen=30)
    #requests_made = 0
    #start_time = time.time()

    filename = 'palm_aug_cln_'+dset+'.csv'

    # Determine where to start
    if os.path.exists(filename):
        df_saved = pd.read_csv(filename)
        start_index = int(len(df_saved)+1)#int(df_saved['counter'].max() + 1)
    else:
        start_index = 0

    #val_df = val_df.reset_index(drop=True)
    # Iterate through the validation set
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)

        # If file is empty, write the header
        if f.tell() == 0:
            writer.writerow(['id', 'input_text', 'predicted_label', 'text_response', 'true_label'])

        for i in range(start_index, len(ds)):#len(val_ds)):  
            row = ds[i]#.iloc[i]#val_ds[i]#
            text = row["text"]

            prompt = f"Given the following tweet, determine if it refers to a real disaster, responding only with 'yes' if it does and 'no' if it doesn't: \"{text}\""
            prefix = ""

            badctr = 0
            while True:
                if len(queue) == 30:
                    time_since_first_request = time.time() - queue[0]
                    if time_since_first_request < 60:  # If less than a minute has passed
                        time.sleep(60 - time_since_first_request)  # Sleep the remaining seconds

                completion = palm.generate_text(
                model=model,
                prompt=prefix + prompt,
                temperature=0,
                max_output_tokens=3,
                safety_settings=[
                    {
                        "category": safety_types.HarmCategory.HARM_CATEGORY_TOXICITY,
                        "threshold": safety_types.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
                    },
                    {
                        "category": safety_types.HarmCategory.HARM_CATEGORY_SEXUAL,
                        "threshold": safety_types.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
                    },
                    {
                        "category": safety_types.HarmCategory.HARM_CATEGORY_VIOLENCE,
                        "threshold": safety_types.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
                    },
                    {
                        "category": safety_types.HarmCategory.HARM_CATEGORY_DANGEROUS,
                        "threshold": safety_types.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
                    },
                    {
                        "category": safety_types.HarmCategory.HARM_CATEGORY_DEROGATORY,
                        "threshold": safety_types.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
                    },
                    {
                        "category": safety_types.HarmCategory.HARM_CATEGORY_VIOLENCE,
                        "threshold": safety_types.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
                    },
                    {
                        "category": safety_types.HarmCategory.HARM_CATEGORY_UNSPECIFIED,
                        "threshold": safety_types.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
                    },
                ]   
                )
                """
                requests_made += 1
                if requests_made == 30:
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    if elapsed_time < 60:  # If less than a minute has passed
                        time.sleep(60 - elapsed_time)  # Sleep the remaining seconds

                    # Reset the count and the start time
                    requests_made = 0
                    start_time = time.time()
                """
                queue.append(time.time())

                print("returnval=",completion.result)
                if completion.result != None:
                    answer = str(completion.result).strip().lower()
                    print("good")
                    prefix = ""
                    break
                else:
                    print("bad")
                    print("prompt=",prompt)
                    badctr += 1
                    print("badctr=",badctr)
                    if badctr == 3:
                        answer = "None"
                        break
                    prefix = "Try again: "
                    
            
            if answer == "yes":
                predicted_label = 1
            elif answer == "no":
                predicted_label = 0
    #        else:
    #            predicted_label = -1
            if "eval" in dset:
                true_label = row["target"]
                if predicted_label == true_label:
                    correct_predictions += 1
                #print("correct_predictions=", correct_predictions)

            # Store the text response along with the input text and predicted label
            #responses.append({
            #    "input_text": text,
            #    "predicted_label": predicted_label,
            #    "text_response": answer,
            #    "true label": true_label
            #})
            writer.writerow([row["id"], text, predicted_label, answer, true_label])
            print("index=", i)
            time_since_first_request = time.time() - queue[0]
            if time_since_first_request < 60:  # If less than a minute has passed
                time.sleep(60 - time_since_first_request)  # Sleep the remaining seconds

        #    if index == 1:
        #        break

    # Calculate and print the accuracy
    accuracy = correct_predictions / len(ds)
    print(f"Correct predictions: {correct_predictions}")
    print(f"Total predictions: {len(ds)}")
    print(f"Accuracy: {accuracy * 100:.2f}%")

# Save the responses to a file
#responses_df = pd.DataFrame(responses)
#responses_df.to_csv("responses_palm_aug_eval.csv", index=False)

