## Dataset Source

Dataset Link:
https://drive.google.com/drive/folders/1akV6po4Nrgkc3yQrJkzA6cJlV-wBvUYs?usp=sharing


## Task 1: Dataset Understanding

#### Number of Records

The dataset contains 1500 records and 6 columns. Each record represents a customer support message along with its associated sentiment label and other related information.

#### Target Labels / Classes

The target column used for classification is sentiment_label.  
The dataset contains three sentiment classes:

- positive
- neutral
- negative

#### Sample Text Records

Sample customer messages are displayed to understand the structure and content of the dataset. The messages include customer queries, complaints, refund issues, and account-related requests.

#### Average Text Length

The average text length of customer messages is approximately 72.76 characters. This helps in understanding the typical size of input text before preprocessing and sequence modeling.

#### Class Distribution

The sentiment classes are distributed as follows:

- neutral : 524
- negative : 497
- positive : 479

The dataset is fairly balanced because all three classes contain a similar number of records.

## Task 2: Text Preprocessing

#### Lowercasing
All customer messages are converted into lowercase format to maintain consistency during NLP preprocessing.

#### Removing Special Characters
Unnecessary symbols and special characters are removed from the text while preserving meaningful numerical values such as ticket numbers.

#### Tokenization
The cleaned text is divided into individual words called tokens using NLTK word tokenization.

#### Stopwords Removal
Common English stopwords such as is, the, about, and are removed because they do not contribute significant meaning for text classification.

#### Padding Sequences
Text sequences are converted into numerical sequences and padded to equal length using pad_sequences(). This ensures consistent input size for sequence-based models like RNN and LSTM.

## Task 3: Text Vectorization

#### TF-IDF Vectorization
Text data is converted into numerical format using TF-IDF Vectorization with TfidfVectorizer from Scikit-learn.

The TF-IDF matrix shape is:

1500 rows representing customer messages
667 columns representing unique text features or words

#### Why Text Must Be Converted into Vectors
Machine learning and deep learning models cannot process raw text directly. Text data must be transformed into numerical vectors so that mathematical computations and pattern recognition can be performed by the model. TF-IDF assigns numerical importance scores to words based on their frequency within individual documents and across the dataset.

## Task 4: Baseline Model

#### Logistic Regression with TF-IDF
A baseline text classification model is built using Logistic Regression with TF-IDF vectorized text features.
The dataset is divided into training and testing sets using train_test_split() with 80% training data and 20% testing data.

#### Model Evaluation
The model is evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

The classification report provided precision, recall, and F1-score values for all sentiment classes:
- positive
- neutral
- negative

A confusion matrix visualization is generated to display the classification performance of the model on the testing dataset. The model achieved an overall accuracy score of 1.00. The model achieved 100% accuracy on the test dataset. This may indicate that the dataset is highly structured or contains easily separable sentiment patterns.

## Task 5: Sequence Model – LSTM Architecture
The Sequential LSTM architecture consisted of:
- An Embedding layer for converting tokenized input sequences into dense word vectors.
- An LSTM layer for learning sequential dependencies and contextual information from text data.
- A Dense output layer with softmax activation for multi-class sentiment classification.

The model summary displayed the layer structure, output shapes, and trainable parameters of the sequence model architecture.

#### Input Sequence
Customer messages are converted into numerical token sequences using a tokenizer. These sequences are used as input data for the LSTM model.

#### Embedding Layer
The Embedding layer transformed integer token sequences into dense vector representations, helping the model learn semantic relationships between words.

#### Recurrent/Sequence Layer
The LSTM layer processed the input sequence step-by-step and captured sequential patterns and contextual dependencies present in customer messages.

#### Output Layer
The Dense output layer with softmax activation predicted the probability of the three sentiment classes:
positive, neutral, and negative.

#### Loss Function
The model used sparse_categorical_crossentropy as the loss function because the sentiment labels are multi-class categorical values encoded as integers.

#### Evaluation Metric
Accuracy is used as the evaluation metric to measure the classification performance of the LSTM model.

## Task 6: Attention and Transformer Reflection

#### Why RNNs Struggle with Long-Term Dependencies
RNNs process text sequentially and often struggle to retain information from earlier time steps when handling long sequences. This creates difficulty in learning long-term dependencies within text data.

#### How LSTMs Help with Memory
LSTMs use memory cells and gating mechanisms to preserve important information for longer durations. This helps to reduce the vanishing gradient problem commonly found in traditional RNNs.

#### What Attention Solves in Sequence-to-Sequence Tasks
Attention mechanisms enable the model to focus on the most relevant parts of the input sequence while generating outputs. This improves performance in tasks such as machine translation and text summarization.

#### Why Transformers Are Important in Modern NLP and Generative AI
Transformers process entire sequences in parallel using self-attention mechanisms to capture contextual relationships efficiently. They form the foundation of modern NLP and Generative AI models such as GPT and BERT.

## Author

Shinoj K R