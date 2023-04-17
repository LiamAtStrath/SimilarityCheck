import argparse
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize
from nltk.metrics import jaccard_distance
from pdfminer.high_level import extract_text

def preprocess(text):
    tokens=word_tokenize(text.lower())
    stop_words=set(stopwords.words('english'))
    tokens=[token for token in tokens if token not in stop_words]
    stemmer=PorterStemmer()
    tokens=[stemmer.stem(token) for token in tokens]
    lemmatizer=nltk.WordNetLemmatizer()
    tokens=[lemmatizer.lemmatize(token,get_wordnet_pos(token)) for token in tokens]
    preprocessed_text=' '.join(tokens)
    return preprocessed_text

def get_wordnet_pos(token):
    tag=nltk.pos_tag([token])[0][1][0].upper()
    tag_dict={"J": wn.ADJ,"N": wn.NOUN,"V": wn.VERB,"R": wn.ADV}
    return tag_dict.get(tag,wn.NOUN)

def extract_pdf_text(filename):
    with open(filename,'rb')as f:
        text=extract_text(f)
    return text.strip()

def compare_similarity(doc1,doc2):
    preprocessed_doc1=preprocess(doc1)
    preprocessed_doc2=preprocess(doc2)
    sentences_doc1=sent_tokenize(preprocessed_doc1)
    sentences_doc2=sent_tokenize(preprocessed_doc2)
    if len(sentences_doc1)==0 or len(sentences_doc2)==0:
        return 0,[]
    similarity_scores={}
    for i,sentence1 in enumerate(sentences_doc1):
        for j,sentence2 in enumerate(sentences_doc2):
            sentence1_tokens=set(word_tokenize(sentence1))
            sentence2_tokens=set(word_tokenize(sentence2))
            jaccard_score=1-jaccard_distance(sentence1_tokens,sentence2_tokens)
            similarity_scores[(i,j)]=jaccard_score
    sorted_scores=sorted(similarity_scores.items(),key=lambda x:x[1],reverse=True)
    most_similar_sentences=[]
    for (i,j),score in sorted_scores:
        if i not in [x[0] for x in most_similar_sentences] and j not in [x[1] for x in most_similar_sentences]:
            most_similar_sentences.append(((i,j),score))
            if len(most_similar_sentences)==min(len(sentences_doc1),len(sentences_doc2)):
                break
    avg_similarity_score=sum(similarity_scores.values())/len(similarity_scores)
    return avg_similarity_score,most_similar_sentences

parser=argparse.ArgumentParser(description='Compare the similarity between two PDF documents.')
parser.add_argument('doc1',metavar='doc1.pdf',type=str,help='the first PDF document')
parser.add_argument('doc2',metavar='doc2.pdf',type=str,help='the second PDF document')
args=parser.parse_args()

doc1_text=extract_pdf_text(args.doc1)
doc2_text=extract_pdf_text(args.doc2)
similarity_score,most_similar_sentences=compare_similarity(doc1_text,doc2_text)

print('The similarity score is:',similarity_score)
print('The most similar sentences and their similarity scores are:')
for (i,j),score in most_similar_sentences:
    print(f'Document 1 sentence {i}: {sent_tokenize(doc1_text)[i]}')
    print(f'Document 2 sentence {j}: {sent_tokenize(doc2_text)[j]}')
    print(f'Similarity score: {score}')
    print()