import csv
from allennlp.predictors.predictor import Predictor
import allennlp_models.rc

# load predictor
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/bidaf-model-2020.03.19.tar.gz")

# load abstracts
result_dicts = []
with open('data/input/clean_docs_50_complete_share.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if 'include' in row['include/exclude']:
            result = dict()
            result['pmid'] = row['pmid']
            result['abstract'] = row['abstract']
            sentences = result['abstract']
            sentences = sentences.split('.')
            result['sentences'] = sentences
            result['drug'] = row['drugs']
            result['result'] = row['association']
            result_dicts.append(result)

questions = ['What was the impact of drug on the cancer/tumor?',
             'Did drug increase the cancer, decrease the cancer, or have no effect on the cancer?',
             'What were the results, or what was shown about drug and the tumor/cancer?']

def format_question(original, drug):
    return original.replace('drug', drug)

# get answers
for result_dict in result_dicts:
    answers = []
    for q in questions:
        question = format_question(q, result_dict['drug'])
        pdct = predictor.predict(passage=result_dict['abstract'], question=question)
        prediction = pdct['best_span_str']
        prediction_sents = prediction.split('.')

        original_sents = []
        for ps in prediction_sents:
            for s in result_dict['sentences']:
                if ps in s:
                    original_sents.append(s)

        answer = ' '.join(original_sents)
        answers.append((question, answer))
    result_dict['qa_pairs'] = answers

# write each qa pair to a new row
with open('data/output/qa_test_bidaf_2.csv', 'w', newline='') as csvfile:
    fieldnames = ['pmid', 'question', 'answer', 'drug']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for result_dict in result_dicts:
        for (q, a) in result_dict['qa_pairs']:
            writer.writerow({'pmid': result_dict['pmid'], 'question': q, 'answer': a, 'drug': result_dict['drug']})


