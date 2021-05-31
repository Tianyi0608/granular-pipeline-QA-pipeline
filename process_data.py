import json
from nltk import sent_tokenize
import re
import tokenizations
import questions as Q
import csv
import random as rd
import argparse

def read_data_from_bpjson(file_path):
    with open(file_path, 'r') as f:
        data_file = {}
        entries = json.load(f)["entries"]
        # key_set = set()
        for index, doc_id in enumerate(entries.keys()):
            # format: {'doc-i':{"segment_text":'XXX',"event_dict":{},"span_dict":{},"sentence_dict":{},"template_dict":{}}...}
            data_file['doc-' + str(index)] = {}

            segment_text = entries[doc_id]["segment-text"]
            data_file['doc-' + str(index)]["segment_text"] = segment_text

            event_dict = {}
            events = entries[doc_id]["annotation-sets"]["basic-events"]["events"]
            for event_id in events.keys():
                event_dict[event_id] = {}
                event_dict[event_id]["anchor_ssid"] = events[event_id]["anchors"]
                event_dict[event_id]["event_type"] = events[event_id]["event-type"]
            data_file['doc-' + str(index)]["event_dict"] = event_dict

            # get the index of each span
            span_sets = entries[doc_id]["annotation-sets"]["basic-events"]["span-sets"]
            span_dict = {}  # {"ss-1":[(start,end,'string')...],...}
            for span_id in span_sets.keys():
                # "string" instead "hstring"
                span_dict[span_id] = [(span["start"], span["end"], span["string"]) for span in span_sets[span_id]["spans"]]
            data_file['doc-' + str(index)]["span_dict"] = span_dict

            # get the index of each sentence
            sentence_dict = {}  # {"0":(0,41),"1":(43,59)...}
            sentences = sent_tokenize(segment_text)
            sentences = [sent.strip() for segment in sentences for sent in re.split(r'[\f\n\r\t\v]', segment) if
                         not sent in ['', ' ']]
            # print(sentences)
            sent_spans = tokenizations.get_original_spans(sentences, segment_text)
            for i, (s, e) in enumerate(sent_spans):
                sentence_dict[str(i)] = (s, e)
            data_file['doc-' + str(index)]["sentence_dict"] = sentence_dict

            # template_dict format: {"template-1":{"anchor_ssid":"ss-x","template-type":"XXX","fields":{"when":["ss-x","ss-x"]...}}...}
            template_dict = {}
            # in some cases, there is no granular-templates
            if "granular-templates" in entries[doc_id]["annotation-sets"]["basic-events"].keys():
                granular_templates = entries[doc_id]["annotation-sets"]["basic-events"]["granular-templates"]
                for template_id in granular_templates.keys():
                    template_dict[template_id] = {}
                    template_dict[template_id]["anchor_ssid"] = granular_templates[template_id]["template-anchor"]
                    template_dict[template_id]["template_type"] = granular_templates[template_id]["template-type"]
                    template_dict[template_id]["fields"] = {}
                    for key in granular_templates[template_id].keys():
                        if key not in ["template-anchor", "template-id", "template-type"]:
                            # key_set.add(key)
                            # print(granular_templates[template_id][key])
                            if not isinstance(granular_templates[template_id][key],list):  # some are boolean value or str value
                                template_dict[template_id]["fields"][key] = granular_templates[template_id][key]
                            else:
                                template_dict[template_id]["fields"][key] = [item["ssid"]
                                                                             if "ssid" in granular_templates[template_id][key][i].keys()
                                                                             else item["event-id"]
                                                                             for i, item in enumerate(granular_templates[template_id][key])]
            data_file['doc-' + str(index)]["template_dict"] = template_dict

    # print(key_set)
    return data_file

def convert_data_to_doc_template_file(data_file, doc_template_file_path):
    doc_template_dict = {}
    # total_ques,no_answer_ques=0,0
    number_list = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
                   'eleven', 'twelve', 'thirteen', 'fifteen', 'twenty', 'thirty', 'forty', 'fifty',
                   'dozen', 'dozens', 'hundred', 'thousand', 'million']
    # total_ques,no_answer_ques=0,0
    for doc_id in data_file.keys():
        doc_template_dict[doc_id] = {}
        for template_id in data_file[doc_id]["template_dict"].keys():
            doc_template_dict[doc_id][template_id] = {}

            template_type = data_file[doc_id]["template_dict"][template_id]["template_type"]
            doc_template_dict[doc_id][template_id]["template_type"] = template_type
            template_answer_index = []  # used to identify the input sentences

            doc_template_dict[doc_id][template_id]["qa_pairs"] = []  # [{"question":xxx,"answer":[{},...]},{...},...]

            # add all no-answers
            # print(Q.template_keys[template_type]-data_file[doc_id]["template_dict"][template_id]["fields"].keys())

            for key in Q.template_keys[template_type] - data_file[doc_id]["template_dict"][template_id][
                "fields"].keys():
                questions = Q.templates[template_type][key]
                for question in questions:
                    doc_template_dict[doc_id][template_id]["qa_pairs"].append({"question": question, "answer": []})

            # total_ques+=len(Q.template_keys[template_type])
            # no_answer_ques+=len(Q.template_keys[template_type]-data_file[doc_id]["template_dict"][template_id]["fields"].keys())

            # add question-answer pairs
            old_new_index_map = {}
            for key in data_file[doc_id]["template_dict"][template_id]["fields"].keys():
                # extract all the keys, so need to filter those has string or boolean value
                if isinstance(data_file[doc_id]["template_dict"][template_id]["fields"][key], list):
                    answer = []
                    # for every key in fields, generate the question and the answer
                    for item in data_file[doc_id]["template_dict"][template_id]["fields"][key]:
                        if item[:3] == 'ss-':
                            for span_item in data_file[doc_id]["span_dict"][item]:
                                start, end, string = span_item
                                # answer.append({"text": string, "start": start, "end": end})
                                answer.append({"text": string})
                                old_new_index_map[start] = -1
                                old_new_index_map[end] = -1
                                template_answer_index.append((start, end))
                        elif item[:6] == 'event-':
                            ssid = data_file[doc_id]["event_dict"][item]["anchor_ssid"]
                            for span_item in data_file[doc_id]["span_dict"][ssid]:
                                start, end, string = span_item
                                # answer.append({"text": string, "start": start, "end": end})
                                answer.append({"text": string})
                                old_new_index_map[start] = -1
                                old_new_index_map[end] = -1
                                template_answer_index.append((start, end))
                        else:
                            print('need an item that can be matched to a ssid', item)
                    if key in Q.keys:  # don't need protest-event,corrupt-event,... now
                        questions = Q.templates[template_type][key]  # a list
                    else:
                        questions = ''
                    if questions != '':
                        if len(questions) > 1:
                            answer_dict = {}
                            for q in questions:
                                if q.split()[0] == 'Who':
                                    answer_dict[q] = []
                                    for i in range(len(answer) - 1, -1, -1):
                                        if not (bool(re.search(r'\d', answer[i]["text"])) or any(
                                                [num in answer[i]["text"].lower() for num in number_list])):
                                            # print(answer[i])
                                            answer_dict[q].append(answer.pop(i))
                                if q.split()[:2] == ['How', 'many']:
                                    answer_dict[q] = []
                                    for i in range(len(answer) - 1, -1, -1):
                                        if (bool(re.search(r'\d', answer[i]["text"])) or any(
                                                [num in answer[i]["text"].lower() for num in number_list])):
                                            answer_dict[q].append(answer.pop(i))
                                # some q don't have any answer, all the candidate answers go into another question.
                                # if len(answer_dict[q])==0:
                                # print(q)

                            # if len(answer)!=0:
                            #     print('answers not match questions: ',questions,answer)
                            for q, a in answer_dict.items():
                                # # delete no answer
                                # if a != []:
                                doc_template_dict[doc_id][template_id]["qa_pairs"].append({"question": q, "answer": a})
                        else:
                            doc_template_dict[doc_id][template_id]["qa_pairs"].append(
                                {"question": questions[0], "answer": answer})

            # get all the sentences needed for an input
            sent_index = set()
            for (answer_s, answer_e) in template_answer_index:
                for sent_id in data_file[doc_id]["sentence_dict"].keys():
                    (sent_s, sent_e) = data_file[doc_id]["sentence_dict"][sent_id]
                    if sent_s <= answer_s and answer_s <= sent_e:
                        # sent_index.add(int(sent_id))
                        sent_index_s = int(sent_id)
                        old_new_index_map[answer_s] = (int(sent_id), answer_s - sent_s)
                    if sent_s <= answer_e and answer_e <= sent_e:
                        # sent_index.add(int(sent_id))
                        sent_index_e = int(sent_id)
                        old_new_index_map[answer_e] = (int(sent_id), answer_e - sent_s)
                    if old_new_index_map[answer_s] != -1 and old_new_index_map[answer_e] != -1:
                        for i in range(sent_index_s, sent_index_e + 1):
                            sent_index.add(i)
                        break
            sent_index = list(sent_index)
            sent_index.sort()

            # # get the new start index for each sentence
            # sent_start_index = {}
            # total_length = 0
            # for sent_id in sent_index:
            #     sent_start_index[sent_id] = total_length
            #     total_length += data_file[doc_id]["sentence_dict"][str(sent_id)][1] - \
            #                     data_file[doc_id]["sentence_dict"][str(sent_id)][0] + 1  # e-s+space
            # # update old_new_index_map
            # for old_index in old_new_index_map.keys():
            #     (sent_id, addition) = old_new_index_map[old_index]
            #     old_new_index_map[old_index] = sent_start_index[sent_id] + addition
            # # print(old_new_index_map)
            # doc_template_dict[doc_id][template_id]["index_map"] = old_new_index_map

            # get the sentences
            sent = []
            for sent_id in sent_index:
                (s, e) = data_file[doc_id]["sentence_dict"][str(sent_id)]
                sent.append(data_file[doc_id]["segment_text"][s:e])
            doc_template_dict[doc_id][template_id]["sentences"] = ' '.join(sent)
            # print(len(doc_template_dict[doc_id][template_id]["sentences"].split()))

    # print(no_answer_ques/total_ques)

    # add question_id for each template
    # num_ques=0
    # for doc_id in doc_template_dict.keys():
    #     for template_id in doc_template_dict[doc_id].keys():
    #         start=num_ques
    #         num_ques+=len(doc_template_dict[doc_id][template_id]["qa_pairs"])
    #         end=num_ques
    #         doc_template_dict[doc_id][template_id]["question_id_s_e"]=(start,end)

    # format: {'doc-i':{'template-j':{'sentences':xxx, 'qa_pairs':{}, 'question_id_s_e':(s,e)}...}...}
    with open(doc_template_file_path,'w') as f_doc_temp:
        json.dump(doc_template_dict, f_doc_temp, indent=4)
    # return doc_template_dict

def convert_to_QA_binary(doc_template_file_path, csv_input_file_path, csv_answer_file_path):

    with open(doc_template_file_path,'r') as f_doc_temp:
        doc_template_dict=json.load(f_doc_temp)

    with open(csv_input_file_path, 'w', newline='') as f:
        # binary:
        # fieldnames = ['question', 'sentence', 'label']
        # binary_reverse:
        # fieldnames = ['sentence', 'question', 'label']
        # binary_wnli:
        fieldnames = ['sentence1', 'sentence2', 'label']
        csv_writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=',',quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writeheader()
        question_id = 0
        no_answer,has_answer=0,0
        for doc_id in doc_template_dict.keys():
            for template_id in doc_template_dict[doc_id].keys():
                sentence = doc_template_dict[doc_id][template_id]["sentences"]
                for qa_pairs in doc_template_dict[doc_id][template_id]["qa_pairs"]:  # qa_pairs: {"question":xxx,"answer":[{},...]}
                    question_id += 1
                    question = qa_pairs["question"]
                    answers = []
                    for answer_dict in qa_pairs["answer"]:
                        # {"text": string}
                        answer = answer_dict["text"]
                        answers.append(answer)
                    if answers==[]:
                        no_answer+=1
                        # csv_writer.writerow({'question': question, 'sentence': sentence, 'label': 'no-answer'})
                        csv_writer.writerow({'sentence1': sentence, 'sentence2': question, 'label': 0})
                    else:
                        has_answer += 1
                        # csv_writer.writerow({'question': question, 'sentence': sentence, 'label': 'has-answer'})
                        csv_writer.writerow({'sentence1': sentence, 'sentence2': question, 'label': 2})
        # print(no_answer,has_answer,question_id)
        # print(total_ques,no_answer_ques)

    with open(csv_answer_file_path, 'w', newline='') as f:
        fieldnames = ['sentence1', 'sentence2', 'answer']
        csv_writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=',',quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writeheader()
        for doc_id in doc_template_dict.keys():
            for template_id in doc_template_dict[doc_id].keys():
                sentence = doc_template_dict[doc_id][template_id]["sentences"]
                for qa_pairs in doc_template_dict[doc_id][template_id]["qa_pairs"]:  # qa_pairs: {"question":xxx,"answer":[{},...]}
                    question_id += 1
                    question = qa_pairs["question"]
                    answers = []
                    for answer_dict in qa_pairs["answer"]:
                        # {"text": string}
                        answer = answer_dict["text"]
                        answers.append(answer)
                    if answers==[]:
                        csv_writer.writerow({'sentence1': sentence, 'sentence2': question, 'answer': 'no-answer'})
                    else:
                        csv_writer.writerow({'sentence1': sentence, 'sentence2': question, 'answer': '|'.join(answers)})

# convert_bert_to_better_format('../data/template_qa/test_has_answer.json', '../models/roberta-large_has_answer_string/nbest_predictions_.json', '../convert_to_bpjson_has_answer.json', '../data/bp_json/granular.eng-provided-72.0pct.devtest-15.0pct.ref.d.bp.json')

def calculate_f1(gold_file_path,pred_file_path,csv_pred_file_path):
    question_id_label_dict = {}
    with open(csv_pred_file_path) as f_pred:
        csv_reader = csv.reader(f_pred, delimiter='\t')
        for row_id, row in enumerate(csv_reader):
            if row_id == 0:
                continue
            question_id, pred_label = str(int(row[0])+1), int(row[1])
            # print(type(question_id),type(pred_label))
            # print(question_id,pred_label)
            question_id_label_dict[question_id] = pred_label

    with open(pred_file_path, 'r') as f_pred:
        pred = json.load(f_pred)
        for question_id in pred.keys():
            if question_id_label_dict[question_id]==0:
                pred[question_id]=""

    gold_answer_dict={}
    with open(gold_file_path, 'r') as f_gold:
        gold = json.load(f_gold)["data"]  # a list
        for doc in gold:
            for temp in doc["paragraphs"]:  # a list
                # field_set=set()
                for qa in temp["qas"]:
                    # gold_q = qa["question"]
                    gold_a = qa["answers"]
                    id = qa["id"]
                    gold_answers=[a["text"] for a in gold_a] if gold_a!=[] else []
                    gold_answer_dict[id]=gold_answers

    has_answer_num_same=0
    no_answer_num_same=0
    has_answer_true_total=0
    no_answer_true_total=0
    has_answer_pred_total = 0
    no_answer_pred_total = 0
    for question_id in gold_answer_dict.keys():
        # count pred has/no answer
        if pred[question_id]=='':
            no_answer_pred_total+=1
        else:
            has_answer_pred_total+=1

        # count true has/no answer and same
        if gold_answer_dict[question_id]==[]:
            no_answer_true_total+=1
            if pred[question_id]=='':
                no_answer_num_same+=1
        else:
            has_answer_true_total+=1
            for gold_a in gold_answer_dict[question_id]:
                if gold_a in pred[question_id]: # the gold_a string is included in pred string
                    # print(gold_a,pred[key])
                    has_answer_num_same+=1
                    break
    print('type precision recall f1')
    precision = 1.0 * no_answer_num_same / no_answer_pred_total
    recall = 1.0 * no_answer_num_same / no_answer_true_total
    f1 = (2 * precision * recall) / (precision + recall)
    print('no-answer',precision, recall, f1)

    precision = 1.0 * has_answer_num_same / has_answer_pred_total
    recall = 1.0 * has_answer_num_same / has_answer_true_total
    f1 = (2 * precision * recall) / (precision + recall)
    print('has-answer', precision, recall, f1)

    precision = 1.0 * (no_answer_num_same+has_answer_num_same) / len(pred)
    recall = 1.0 * (no_answer_num_same+has_answer_num_same) / len(gold_answer_dict)
    f1 = (2 * precision * recall) / (precision + recall)
    print('total', precision, recall, f1)

# the conversion from has/no-answer binary classification to has-answer argument extraction
def convert_to_squad_json(doc_template_file_path, squad_input_file_path):
    # question_id_label_answer_dict={}
    # with open(csv_pred_file_path) as f_pred:
    #     csv_reader = csv.reader(f_pred, delimiter='\t')
    #     for row_id, row in enumerate(csv_reader):
    #         if row_id == 0:
    #             continue
    #         question_id, pred_label = int(row[0]), int(row[1]) # start from 0
    #         # print(type(question_id),type(pred_label))
    #         # print(question_id,pred_label)
    #         question_id_label_answer_dict[question_id]["pred_label"]=pred_label

    # with open(csv_answer_file_path) as f_ans:
    #     csv_reader = csv.reader(f_ans, delimiter=',')
    #     for row_id,row in enumerate(csv_reader):
    #         if row_id==0:
    #             continue
    #         sentence,question,answer=row
    #         answer_list=answer.split('|')
    #         question_id_label_answer_dict[row_id - 1]["question"]=question
    #         question_id_label_answer_dict[row_id-1]["answer"]=answer_list

    with open(doc_template_file_path,'r') as f_doc_temp:
        doc_template_dict=json.load(f_doc_temp)

    question_id=0
    squad_dict = {"version": "train", "data": [{"title": doc_id, "paragraphs": []} for doc_id in doc_template_dict.keys()]}
    for doc_index,doc_id in enumerate(doc_template_dict.keys()):
        for template_id in doc_template_dict[doc_id].keys():
            template_unit = {}  # {"qas":[],"context":"XXX"}
            template_unit["qas"] = []
            template_unit["context"] = doc_template_dict[doc_id][template_id]["sentences"]
            for qa_pairs in doc_template_dict[doc_id][template_id]["qa_pairs"]: # qa_pairs: {"question":xxx,"answer":[{},...]}
                question_id += 1
                question = qa_pairs["question"]
                answers = [{"text": answer["text"]} for answer in qa_pairs["answer"]]
                template_unit["qas"].append({"question": question, "answers": answers, "id": str(question_id),"is_impossible": answers == []})
            squad_dict["data"][doc_index]["paragraphs"].append(template_unit)

    with open(squad_input_file_path,'w') as f_squad:
        json.dump(squad_dict, f_squad, indent=4)

# data_file=read_data_from_bpjson('../data/bp_json/granular.eng-provided-72.0pct.devtest-15.0pct.ref.d.bp.json')
# convert_data_to_QA_binary(data_file,'./data/csv_input.csv', './data/csv_answer.csv','./data/doc_template.json')
# convert_to_squad_json('./data/doc_template.json', './data/squad_input.json')

def convert_bert_to_better_format(squad_true_path, squad_pred_path, csv_pred_file_path, bpjson_path, ref_path):
    with open(ref_path, 'r') as f:
        entries = json.load(f)['entries']

    with open(squad_true_path, 'r') as f_true:
        gold=json.load(f_true)["data"]
        text_dict={}
        for doc_idx in range(len(gold)):
            templates_gold = gold[doc_idx]["paragraphs"]
            for temp_idx in range(len(templates_gold)):
                for ques_idx in range(len(templates_gold[temp_idx]['qas'])):
                    ques_id=templates_gold[temp_idx]['qas'][ques_idx]['id']
                    text_dict[ques_id]=templates_gold[temp_idx]['context'] # pred_temp_idx start from 1
    # print(text_dict.keys())

    question_id_label_dict = {}
    with open(csv_pred_file_path) as f_pred:
        csv_reader = csv.reader(f_pred, delimiter='\t')
        for row_id, row in enumerate(csv_reader):
            if row_id == 0:
                continue
            question_id, pred_label = str(int(row[0]) + 1), int(row[1])
            # print(type(question_id),type(pred_label))
            # print(question_id,pred_label)
            question_id_label_dict[question_id] = pred_label

    with open(squad_pred_path,'r') as f_pred:
        preds=json.load(f_pred)
        # process predict no-answer
        for question_id in preds.keys():
            if question_id_label_dict[question_id] == 0:
                preds[question_id] = [{'text': '', 'probability': 1., 'start_logit': 1., 'end_logit': 1.}]

        for id in preds.keys():
            # process ''+'empty'
            null = {'text': '', 'probability': 0., 'start_logit': 0., 'end_logit': 0.}
            for i in range(len(preds[id]) - 1, -1, -1):
                if preds[id][i]['text'] == '' or preds[id][i]['text'] == 'empty':
                    temp = preds[id].pop(i)
                    null['probability'] += temp['probability']
                    null['start_logit'] += temp['start_logit']
                    null['end_logit'] += temp['end_logit']
            if null['probability']!=0:
                preds[id].append(null)
            preds[id] = sorted(preds[id], key=lambda x: x["probability"], reverse=True)

            # delete no-answer when the probability of the highest no-answer is below a threshold
            if preds[id][0]["text"]=='':
                if preds[id][0]["probability"]>0.8:
                    preds[id]=[preds[id][0]]
                    continue
                else:
                    preds[id].pop(0)
            for i in range(len(preds[id])-1,-1,-1):
                if i==0 and preds[id][i]["text"]=='':
                    continue
                if preds[id][i]["probability"]<=0.05 or len(preds[id][i]["text"].split())>10 or (len(preds[id][i]["text"]) < 2 and preds[id][i]["text"].isalnum()==False):
                    preds[id].pop(i)

            # only keep one answer
            # preds[id]=[preds[id][0]]
    # print(len([preds[id][0]['text'] for id in preds if preds[id][0]['text']!='']))


    with open(squad_true_path, 'r') as f_true:
        gold=json.load(f_true)["data"]
        for doc_idx, doc_id in enumerate(entries):
            events={k:v for k,v in entries[doc_id]["annotation-sets"]["basic-events"]["events"].items()}
            entries[doc_id]["annotation-sets"]["basic-events"]["events"] = {}
            span_sets={k:v for k,v in entries[doc_id]["annotation-sets"]["basic-events"]["span-sets"].items()}
            entries[doc_id]["annotation-sets"]["basic-events"]["span-sets"] = {}
            entries[doc_id]["annotation-sets"]["basic-events"]["includes-relations"] = {}
            if "granular-templates" in entries[doc_id]["annotation-sets"]["basic-events"].keys():
                templates_better=entries[doc_id]["annotation-sets"]["basic-events"]["granular-templates"]

                ss_num = 0
                ss_event=0
                templates={}
                new_span_sets_dict={}
                new_events_dict={}
                templates_gold=gold[doc_idx]["paragraphs"]
                # print(len(templates_gold),len(templates_better))
                assert len(templates_gold)==len(templates_better), "template number does not match in squad json file and bp_json ref file"
                for temp_idx,temp_id in enumerate(templates_better):
                    field_set=set()
                    temp_type=templates_better[temp_id]["template-type"]
                    templates[temp_id] = {"template-id":temp_id,"template-type":temp_type}
                    # add other value fields to templates
                    for field in Q.templates_leave[temp_type]:
                        if field in templates_better[temp_id].keys():
                            templates[temp_id][field]=templates_better[temp_id][field]
                    # add template_event to templates
                    field=Q.templates_event[temp_type]
                    if field in templates_better[temp_id].keys():
                        templates[temp_id][field] = []
                        for item in templates_better[temp_id][field]:
                            ss_num += 1
                            ss_event+=1
                            old_event_id=item["event-id"]
                            templates[temp_id][field].append({"event-id": "event-" + str(ss_event)})
                            new_events_dict['event-' + str(ss_event)] = \
                                {"agents": [],
                                 "anchors": "ss-" + str(ss_num),
                                 "event-type": None,
                                 "eventid": 'event-' + str(ss_event),
                                 "patients": [],
                                 "ref-events": [],
                                 "state-of-affairs": False}
                            old_anchor_id=events[old_event_id]["anchors"]
                            new_span_sets_dict['ss-' + str(ss_num)] = span_sets[old_anchor_id]
                            new_span_sets_dict['ss-' + str(ss_num)]["ssid"]="ss-" + str(ss_num)
                            # print(ss_event,ss_num,new_span_sets_dict['ss-' + str(ss_num)])

                    # add template anchor and ssid
                    temp_anchor_ssid=templates_better[temp_id]["template-anchor"]
                    ss_num+=1
                    new_span_sets_dict['ss-'+str(ss_num)]=span_sets[temp_anchor_ssid]
                    new_span_sets_dict['ss-' + str(ss_num)]["ssid"]='ss-'+str(ss_num)
                    templates[temp_id]["template-anchor"]='ss-'+str(ss_num)

                    # get question and answers from gold
                    template=templates_gold[temp_idx] # one template
                    for qa_pairs in template["qas"]:
                        question=qa_pairs["question"]
                        field=Q.templates_question_to_key[temp_type][question]
                        filler=Q.templates_filler[temp_type][field]
                        q_id = qa_pairs["id"]
                        preds_answers=preds[q_id] # a list
                        for pred_answer in preds_answers:
                            if pred_answer["text"]!='':
                                if field not in field_set:
                                    field_set.add(field)
                                    templates[temp_id][field]=[]
                                ss_num+=1
                                if filler == 'entity':
                                    new_span_sets_dict['ss-' + str(ss_num)] = \
                                        {"spans": [
                                            {"end": None,
                                             "hend": None,
                                             "hinferred":True,
                                             "hstart": None,
                                             "hstring": None,
                                             "start": None,
                                             "string": pred_answer["text"],
                                             "synclass": None}
                                        ],
                                            "ssid": "ss-" + str(ss_num)}
                                    templates[temp_id][field].append({"ssid":"ss-" + str(ss_num)})
                                    # print('ss-' + str(ss_num))
                                    # print(field,'|',preds_answers)
                                if filler=='event':
                                    ss_event+=1
                                    new_span_sets_dict['ss-' + str(ss_num)] = \
                                        {"spans": [
                                            {"anchor-string": True,
                                            "end": None,
                                            "hend": None,
                                            "hstart": None,
                                            "hstring": None,
                                            "start": None,
                                            "string": pred_answer["text"],
                                            "synclass": "event-anchor"}
                                        ],
                                        "ssid": "ss-" + str(ss_num)}
                                    new_events_dict['event-'+str(ss_event)]= \
                                        {"agents": [],
                                        "anchors": "ss-" + str(ss_num),
                                        "event-type": None,
                                        "eventid": 'event-'+str(ss_event),
                                        "patients": [],
                                        "ref-events": [],
                                        "state-of-affairs": False
                                        }
                                    templates[temp_id][field].append({"event-id": "event-" + str(ss_event)})

                entries[doc_id]["annotation-sets"]["basic-events"]["granular-templates"]=templates
                entries[doc_id]["annotation-sets"]["basic-events"]["span-sets"]=new_span_sets_dict
                entries[doc_id]["annotation-sets"]["basic-events"]["events"] = new_events_dict

    with open(bpjson_path, 'w') as new_f:
        json.dump({"corpus-id": "Granular English, V1.6, Provided Devtest Ref (Obfuscated)", "entries": entries,
                   "format-type": "bp-corpus",
                   "format-version": "v10"}, new_f, indent=4, separators=(',', ':'))

# convert_bert_to_better_format('./data/squad_input.json','./data/nbest_predictions_.json','./data/test_results_None.txt','./data/convert_to_bpjson_pipeline.json', '../data/bp_json/granular.eng-provided-72.0pct.devtest-15.0pct.ref.d.bp.json')

# doc_template_file_path is predefined
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bp_json_ref_file_path",
        default=None,
        type=str,
    )

    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
    )

    parser.add_argument(
        "--bp_json_sys_file_path",
        default=None,
        type=str,
    )

    parser.add_argument(
        "--pre_processing",
        action="store_true",
    )

    parser.add_argument(
        "--post_processing",
        action="store_true",
    )

    args = parser.parse_args()
    if args.pre_processing:
        data_file=read_data_from_bpjson(args.bp_json_ref_file_path)
        convert_data_to_doc_template_file(data_file, './data/doc_template.json')
        convert_to_QA_binary('./data/doc_template.json', args.data_dir+'/csv_input.csv', args.data_dir+'/csv_answer.csv')
        convert_to_squad_json('./data/doc_template.json', args.data_dir+'/squad_input.json')

    if args.post_processing:
        # print(args.qa_file_path, args.predictions_dir+'/nbest_predictions_.json', args.bp_json_sys_file_path, args.bp_json_ref_file_path)
        calculate_f1(args.data_dir + '/squad_input.json', args.data_dir + '/predictions_.json', args.data_dir + '/test_results_None.txt')
        convert_bert_to_better_format(args.data_dir+'/squad_input.json', args.data_dir+'/nbest_predictions_.json', args.data_dir+'/test_results_None.txt', args.bp_json_sys_file_path, args.bp_json_ref_file_path)

if __name__ == "__main__":
    main()
