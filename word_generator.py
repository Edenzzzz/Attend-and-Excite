import csv
import json
import random
from flair.data import Sentence
from flair.models import SequenceTagger
from utils import write_to_txt

tagger = SequenceTagger.load('pos')

def ann_to_list(annotations: dict):
    #keys: info, licenses, images, annotations
    captions = []
    anns = annotations["annotations"]
    for ann in anns:
        captions.append(ann["caption"])
    return captions
    
def extract_nouns_adjectives(text):
        tagger.predict(text)
        #NNS: plural nouns, NN: singular nouns
        nouns = [token.text.lower() for token in text if token.tag == 'NN'] 
        #JJ: adjectives, JJR: comparative adjectives, JJS: superlative adjectives
        adjectives = [token.text.lower() for token in text if token.tag == 'JJ' ] 
        return nouns, adjectives

def run():
    vocab_json = "mscoco/annotations/captions_train2017.json"
    
    print("Extracting nouns and adjectives from", vocab_json)
    with open(vocab_json, 'r') as f:
        annotations = json.load(f)
    
    captions = ann_to_list(annotations)
    print("Number of captions:", len(captions))
    
    random.shuffle(captions)
    
    nouns = set()
    adjectives = set()

    for caption in captions:
        input = Sentence(caption)
        caption_nouns, caption_adjectives = extract_nouns_adjectives(input)
        nouns.update(caption_nouns)
        adjectives.update(caption_adjectives)
        # if len(nouns) > 1000 and len(adjectives) > 1000:
        #      break
    nouns -= adjectives
    nouns = list(nouns)[:10000]
    adjectives = list(adjectives)[:10000]
    write_to_txt(nouns, 'nouns.txt')
    write_to_txt(adjectives, 'adjectives.txt')

def prompts_to_nouns(prompts_dir: str):
    import pickle as pkl 
    import os
    prompts = pkl.load(open(os.path.join(prompts_dir, "prompts.pkl"), "rb"))
    nouns = []
    for prompt in prompts:
        input = Sentence(prompt)
        caption_nouns, _ = extract_nouns_adjectives(input)
        nouns.append(caption_nouns)
    pkl.dump(nouns, open(os.path.join(prompts_dir, "nouns.pkl"), "wb"))
    
if __name__ == "__main__":
    run()