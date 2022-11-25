# loading libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import random
from spacy.training import Example

# import NLP Packages
import spacy
from wordcloud import WordCloud, STOPWORDS
from spacy.util import minibatch, compounding

df = pd.read_csv('drugsComTrain_raw.tsv', sep = '\t')

def process_review(review):
  processed_token = []
  for token in review.split():
      token = ''.join(e.lower() for e in token if e.isalnum())
      processed_token.append(token)
  return ' '.join(processed_token)

all_drugs = df['drugName'].unique().tolist()

all_drugs = [x.lower() for x in all_drugs]

count = 0
TRAIN_DATA = []
for _, item in df.iterrows():
    ent_dict = {}
    if count < 1000:
        review = process_review(item['review'])
        #Locate drugs and their positions once and add to the visited items.
        visited_items = []
        entities = []
        for token in review.split():
            if token in all_drugs:
                for i in re.finditer(token, review):
                    if token not in visited_items:
                        entity = (i.span()[0], i.span()[1], 'DRUG')
                        visited_items.append(token)
                        entities.append(entity)
        if len(entities) > 0:
            ent_dict['entities'] = entities
            train_item = (review, ent_dict)
            TRAIN_DATA.append(train_item)
            count+=1



n_iter = 10
def train_ner(training_data):
    
    TRAIN_DATA = training_data

    nlp = spacy.blank("en")  # create blank Language class
    print("Created blank 'en' model")
    
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
        # nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe("ner")
        
    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])
            
    nlp.begin_training()
    for itn in range(n_iter):
        random.shuffle(TRAIN_DATA)
        losses = {}
        # batch up the examples using spaCy's minibatch
        # batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
        # for batch in batches:
        for batch in minibatch(TRAIN_DATA,size=compounding(4.0, 32.0, 1.001)):
          for texts, annotations in batch:
            doc1 = nlp.make_doc(texts)
            example = Example.from_dict(doc1, annotations)
            nlp.update([example], drop=0.5, losses=losses)
            # texts, annotations = zip(*batch)
            # nlp.update(
            #     texts,  # batch of texts
            #     annotations,  # batch of annotations
            #     drop=0.5,  # dropout - make it harder to memorise data
            #     losses=losses,
            # )
            # nlp.update(batch)
        print("Losses", losses)
    return nlp

nlp2 = train_ner(TRAIN_DATA)

def extract_drug_entity(text):
    docx =  nlp2(text)
    print(docx.ents)
    result=[]
    # result = [ent.label_ for ent in docx.ents]
    for ent in docx.ents:
        # print('a',ent)
        result.append((str(ent),ent.label_))
    return result


# import pickle
# pickle_out = open("drug.pkl","wb")
# pickle.dump(extract_drug_entity, pickle_out)
# pickle_out.close()