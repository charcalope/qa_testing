import csv
from allennlp.predictors.predictor import Predictor
import allennlp_models.pair_classification

predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/decomposable-attention-elmo-2020.04.09.tar.gz")

result_dicts = []
with open('data/output/qa_test_bidaf_2.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        result = dict()
        result['pmid'] = row['pmid']
        result['question'] = row['question']
        result['answer'] = row['answer']
        result['drug'] = row['drug']
        result_dicts.append(result)

hypotheses = ['Drug improved the outcome or decreased the cancer.', 'Drug had no effect on the outcome or the cancer.']

def format_hyp(original, drug):
    return original.replace('Drug', drug)

for result_dict in result_dicts:
    answers = []
    for hyp in hypotheses:
        hypothesis = format_hyp(hyp, result_dict['drug'])
        result_dict['hypothesis'] = hypothesis
        prediction = predictor.predict(hypothesis=hypothesis, premise=result_dict['answer'])
        probs = prediction['label_probs']
        result_dict['entailment'] = probs[0]
        result_dict['contradiction'] = probs[1]

with open('data/output/entail_test_2.csv', 'w', newline='') as csvfile:
    fieldnames = ['pmid', 'question', 'answer/premise', 'hypothesis', 'entailment', 'contradiction']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for result_dict in result_dicts:
        writer.writerow({'pmid': result_dict['pmid'],
                         'question': result_dict['question'],
                         'answer/premise': result_dict['answer'],
                         'hypothesis': result_dict['hypothesis'],
                         'entailment': result_dict['entailment'],
                         'contradiction': result_dict['contradiction']})