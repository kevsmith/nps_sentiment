#!/usr/bin/env python3

import csv
import spacy
from vader_sentiment.vader_sentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
nlp = spacy.load('en_core_web_sm')

print('Date,Account,Person,Email,ResponseType,SentimentScore,Entities')
with open('all_nps_comments.csv') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
    for row in reader:
        if len(row['NPS Comment']) < 10:
            continue
        score = analyzer.polarity_scores(row['NPS Comment'])
        if score['compound'] > -0.0001:
            continue
        doc = nlp(row['NPS Comment'])
        output = '{},"{}","{}","{}","{}",{}'.format(row['Response Date'], row['Account Name'], row['Full Name'],
                                                    row['Contact Email'], row['NPS Response Type'], score['compound'])
        entities = []
        for ent in doc.ents:
            if ent.text not in entities:
                if ent.label_ in ['ORG', 'PERSON']:
                    entities.append(ent.text)
                if ent.label_ == 'CARDINAL' and len(entities) > 0:
                    entities.append(ent.text)
        out_entities = ''
        if len(entities) > 0:
            out_entities = '|'.join(entities)
        else:
            continue
        print('{},"{}"'.format(output, out_entities))
