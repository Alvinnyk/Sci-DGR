import json
import spacy

# Load data
class LoadData:
    def __init__(self, data_path):
        self.data_path = data_path


    def load(self):
        data = []
        with open(self.data_path, 'r', encoding='utf-8') as file:
            for line in file:
                loaded_line = json.loads(line)
                # only extract 3 features
                filtered_line = {key: loaded_line[key] for key in ['string', 'label', 'label_confidence'] if
                                 key in loaded_line}
                data.append(filtered_line)
        return data

# Data preprocessing
# first choose 'sm' to increase the speed, if the dataset is too large, change it to 'lg'
nlp = spacy.load('en_core_web_sm')
class DataPreprocessing:
    def __init__(self):
        # spacy has stop words dictionary
        self.stop_words = spacy.lang.en.stop_words.STOP_WORDS

    def tokenize(self, text):
        # lemmatization and tokenization
        # process the text with spacy to obtain lemmatized forms of words, filtering out punctuation and whitespace tokens
        return [token.lemma_ for token in nlp(text) if not token.is_punct | token.is_space]

    def preprocess(self, text):
        # lowercase
        text = text.lower()
        # lemma and tokenize
        tokens = self.tokenize(text)
        # delete top words
        tokens = [word for word in tokens if word.isalpha() and word not in self.stop_words]
        return tokens

    def preprocess_documents(self, documents):
        # initialize a new document
        preprocessed_docs = []
        for doc in documents:
            # extract the 'string' to preprocess
            text = doc['string']
            # preprocess
            preprocessed_text = self.preprocess(text)
            # add the new string to the new document
            new_doc = doc.copy()
            new_doc['string'] = ' '.join(preprocessed_text)
            preprocessed_docs.append(new_doc)
        return preprocessed_docs


# Training data
data_path = f'C:/Users/windows/Desktop/semester2/CS4248/project/data/scicite/train.jsonl'
data = LoadData(data_path)
train_data = data.load()
print(train_data[:5])

# Preprocessing
data_preprocessor = DataPreprocessing()
preprocessed_documents = data_preprocessor.preprocess_documents(train_data[:5])

for doc in preprocessed_documents:
    print(doc)