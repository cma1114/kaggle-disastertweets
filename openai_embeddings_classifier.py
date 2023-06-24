import os
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential

os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = os.getenv("OPENAI_API_KEY")

def load_data(sep="[SEP]", aug=True, dset='train'):
    if dset=='train':
        df = pd.read_csv("/Users/christopherackerman/input/train_cln.csv")
    else:
        df = pd.read_csv("/Users/christopherackerman/input/test.csv")
    #add keyword and location
    if aug:
        df['keyword'] = df['keyword'].fillna('unknown')
        df['location'] = df['location'].fillna('unknown')
        df['text'] = df.apply(lambda row: f"{row['text']} {sep} keyword: {row['keyword']} {sep} location: {row['location']}", axis=1)
    return df

#load and split the data
train_df = load_data(aug=False,dset='train')
ds = Dataset.from_pandas(train_df)
ds = ds.train_test_split(test_size=0.05, seed=42)

# Rename 'test' to 'eval' and add the new test dataset
test_df = load_data(aug=False,dset='test')
test_ds = Dataset.from_pandas(test_df)
ds = DatasetDict({'train': ds['train'], 'eval': ds['test'], 'test': test_ds})

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_embedding_with_backoff(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

# loop over the training and test sets
for split in ['train', 'eval', 'test']:
    embeddings = []
    print(f"Processing {split} data...")
    for i, row in enumerate(ds[split]):
        if i % 50 == 0: 
            print(f"In {split} i=",i)
        text = row['text']
        embed = get_embedding_with_backoff(text)
        embeddings.append(embed)
    
    embeddings = np.array(embeddings)
    ds_df = ds[split].to_pandas()  # convert Dataset to pandas DataFrame
    for i in range(embeddings.shape[1]):
        ds_df[f'embedding_{i}'] = embeddings[:, i]  # add each dimension of the embeddings as a new column

    # Create a new Dataset from the DataFrame
    ds[split] = Dataset.from_pandas(ds_df)

    # Save the dataframe without embeddings as csv
    ds_df.drop(columns=[f'embedding_{i}' for i in range(embeddings.shape[1])]).to_csv(f"openaiada_data_{split}.csv", index=False)

    # Save the embeddings as npy
    np.save(f"openaiada_embeddings_{split}.npy", embeddings)


# Loading data and embeddings
for split in ['train', 'eval', 'test']:
    df = pd.read_csv(f'openaiada_data_{split}.csv')

    # Load the numpy file
    embeddings_loaded = np.load(f'openaiada_embeddings_{split}.npy', allow_pickle=True)

    # Convert the embeddings back to a DataFrame
    embeddings_df = pd.DataFrame(embeddings_loaded, columns=[f'embedding_{i}' for i in range(embeddings_loaded.shape[1])])

    # Concatenate the embeddings DataFrame with the original DataFrame
    df = pd.concat([df, embeddings_df], axis=1)

    # Create a new Dataset from the DataFrame
    ds[split] = Dataset.from_pandas(df)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.linear_model import LogisticRegression

embedding_cols = [col for col in ds['train'].column_names if 'embedding' in col]
X_train = np.array([ds["train"][col] for col in embedding_cols]).T
y_train = np.array(ds["train"]["target"])
print(len(y_train))

# Train a Logistic Regression model
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
print(len(y_train))

# After training the Logistic Regression model, predict the labels for the validation set
embedding_cols = [col for col in ds['eval'].column_names if 'embedding' in col]
X_val = np.array([ds["eval"][col] for col in embedding_cols]).T
y_val = np.array(ds["eval"]["target"])
y_val_pred = clf.predict(X_val)
print(len(y_val_pred))

# Create a dataframe
output_df = pd.DataFrame({
    'id': ds['eval']['id'],
    'input_text': ds['eval']["text"],
    'predicted_label': y_val_pred,
    'true_label': y_val
})
output_df.to_csv("openai_ada_embeddings_noaug_cln_logreg_eval_short.csv", index=False)

###now do test set
embedding_cols = [col for col in ds['test'].column_names if 'embedding' in col]
X_test = np.array([ds["test"][col] for col in embedding_cols]).T
y_test_pred = clf.predict(X_test)
print(len(y_test_pred))

# Create a dataframe
output_df = pd.DataFrame({
    'id': ds['test']['id'],
    'input_text': ds['test']["text"],
    'predicted_label': y_test_pred
})
output_df.to_csv("openai_ada_embeddings_noaug_logreg_cln_test.csv", index=False)
