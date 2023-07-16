# kaggle-disastertweets
This project uses [Kaggle's Disaster Tweets NLP competition](https://www.kaggle.com/competitions/nlp-getting-started/overview) as a pretext to explore various transformer-based LLMs, finetuning/training methods, and other aspects of practical NLP problem solving. To simulate real-world contraints on a toy problem, I've chosen to confine myself to spend <$10, using Kaggle's free GPU time. This achieved 84.002% accuracy on the test set. If we ignore the very suspicious 100% accuracy teams and take the highest non-perfect entry (85.044% accuracy at the time of submission) as the real leader, this was good for a tie for 31st out of ~1000 teams on the [leaderboard](https://www.kaggle.com/competitions/nlp-getting-started/leaderboard) and a tie for 3rd among teams with only one submission.

Although numerous other LLMs and training approaches were tried, the combinations represented in the code in this repository and that contributed to the Kaggle submission, chosen to represent a diversity of architectures and approaches that worked well, are as follows:

Finetune encoder with classifier head
*     roberta-large
Finetune decoder with lm head
*     distilgpt2
Finetune encoder-decoder with seq2seqlm head (using lora)
*     flan-t5-large
Train classifier on top of embeddings
*     gpt2-xl (decoder)
*     roberta-large (encoder)
*     openai-ada (decoder; calling api)
Finetune using OpenAI API
*     curie (decoder)
Ask instruction-tuned LM with no fine tuning
*     PaLM (decoder)


The contest description presents the challenge as being discrimiating between tweets about real versus metaphorical disasters, using as an example the tweet "On plus side LOOK AT THE SKY LAST NIGHT IT WAS ABLAZE". At first glance, this seemed relatively easy - humans know even without looking at the accompanying picture that the words probably refer to a sunset, so why should modern large language models, which use language so fluently, struggle with it? Perhaps not being able to access images or other context would add some difficulty, but those cases would be relatively rare.

### Looking at the data

Upon inspecting the training data provided by Kaggle I found that it is very messy - there are duplicated rows (some of which have inconsistent labels), missing values, and nonstandard characters. Worse, when trying to understand the underperformance of early iterations by looking at valitation set data they misclassified, I realized that many of the labels are in fact wrong. It's not obvious what to do about that. First there is the question of how to fix it at scale, and then there is uncertainty about whether the test set data is also mislabeled. If the test set data is clean, then a cleaner training set is desirable. If the test set is mislabeled, and there's some regularity to the mislabeling in both the training and test sets, then perhaps a classifier could be trained to spot that regularity and enhance performance.

### Preprocessing

Preliminary testing suggested that standard string normalization was helpful, and in most cases augmenting the input with the keyword and location fields, even though they were often empty, was beneficial or neutral. I also dropped duplicate rows (except for the first one) and rows with the same text but different labels. Finally, a few classifiers were run on the full training set using 5-fold cross validation, and rows where all the classifiers agreed on a label that was different from the "official" label were changed to have that consensus label. These steps are implemented in [majorityvote.ipynb](./majorityvote.ipynb) and created [train_cln.csv](./input/train_cln.csv), which was used as input to the full system that used a different set of classifiers. This last was motivated by manual error analysis revealing that when classifiers were unanimous it was they that were right and the labeling that was wrong, and led to higher classification accuracy on the training set. More aggressive approaches were possible, including manual cleanup of the entire set, but this one was chosen for simplicity and efficiency.

### Baseline

There were somewhat more negative examples in the training set (57.0%), and even more in the cleaned training set (65.7%) since most of the mislabelings were false positives, leading to 56.9% and 66.4% respective accuracies on the validation set (chosen as a random 25% of the training set) for a degenerate classifier that always used the most common label from the training set. A simple n-gram (beyond bigram wasn't helpful) classifier yielded validation set accuracies of 79.2% and 84.9% on the raw and cleaned training sets, respectively.

### Process

Out of personal interest a number of different models and training regimens were explored, including the ones in this repository, spanning a variety of transformer architectures (encoder only, decoder only, encoder-decoder), model sizes (from 88M parameters in distilgpt2 to XXB parameters in PaLM's text-bison-001), training approches (finetuning, finetuning using LoRA, training a classifier on top of embeddings), and classifiers (although really nothing beats good old logistic regression). With no training at all PaLM achieved 85% validation set accuracy; all the other models achieved between 88% and 91%. After observing that classifier errors were only moderately correlated with each other, I decided to train a classifier to predict the correct label from the various classifier outputs and a few other features, and this combined system achieved 92% accuracy on the validation data; this is what was run on the test set and submited to Kaggle.

### Code 
Some code for fine tuning was borrowed/adapted from open online sources; links are included where applicable. The rest of the code is mine or ChatGPT's.

Code for cleaning the data and evaluating and combining the classifiers is in [majorityvote.ipynb](./majorityvote.ipynb). Code for calling OpenAI's API to get their embeddings and the training and classifying on those is in [openai_embeddings_classifier.py](./openai_embeddings_classifier.py). Code for using OpenAI's API for finetuning and then classifying on that is in [disastertweets_openai_ft.py](./disastertweets_openai_ft.py). Code for calling Google's PaLM API to label the data without training is in [callpalm.py](./callpalm.py). The remainder of the classification approaches are in [disaster-tweets-inference.ipnyb](./disaster-tweets-inference.ipynb).
