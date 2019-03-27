# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run BERT on SQuAD."""

from __future__ import absolute_import, division, print_function

import argparse
import collections
import json
import logging
import math
import os
import random
import sys
import re
import string
from io import open
import datetime
import csv

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset, Subset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForQuestionAnswering, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.tokenization import (BasicTokenizer,
                                                  BertTokenizer,
                                                  whitespace_tokenize)

from tensorboard_api import Tensorboard

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

TINY_DATA_SIZE = 240  # number of examples for debug

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class SquadExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (
            self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.start_position:
            s += ", end_position: %d" % (self.end_position)
        if self.start_position:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

def write_error_analysis(args, gold_file=None):
    """Compare Gold-Standard SQuAD json file to Predictions file and write new error analysis file."""

    # All methods between dashed lines are from the official SQuAD 2.0 eval script and DFP starter code
    # https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
    # ---------------------------------------------------------------------------------------------------------------------
    def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
        if not ground_truths:
            return metric_fn(prediction, '')
        scores_for_ground_truths = []
        for ground_truth in ground_truths:
            score = metric_fn(prediction, ground_truth['text'])  # KML added ['text']
            scores_for_ground_truths.append(score)
        return max(scores_for_ground_truths)
    
    def compute_avna(prediction, ground_truths):
        """Compute answer vs. no-answer accuracy."""
        return float(bool(prediction) == bool(ground_truths))
    
    def normalize_answer(s):
        """Convert to lowercase and remove punctuation, articles and extra whitespace."""
    
        def remove_articles(text):
            regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
            return re.sub(regex, ' ', text)
    
        def white_space_fix(text):
            return ' '.join(text.split())
    
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
    
        def lower(text):
            return text.lower()
    
        return white_space_fix(remove_articles(remove_punc(lower(s))))
    
    def get_tokens(s):
        if not s:
            return []
        return normalize_answer(s).split()
    
    def compute_em(a_gold, a_pred):
        return int(normalize_answer(a_gold) == normalize_answer(a_pred))
    
    def compute_f1(a_gold, a_pred):
        gold_toks = get_tokens(a_gold)
        pred_toks = get_tokens(a_pred)
        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            return int(gold_toks == pred_toks)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1    
    # ---------------------------------------------------------------------------------------------------------------------

    import xlsxwriter
    import pandas as pd

    if not gold_file: gold_file=args.predict_file
    input_prediction_file = os.path.join(args.output_dir, args.time_stamp + "predictions.json")
    output_error_analysis_file = os.path.join(args.output_dir, args.time_stamp + "errors")    
    #output_error_analysis_file = os.path.join(args.output_dir, args.time_stamp + "errors.json")
    #output_error_analysis_filec = os.path.join(args.output_dir, args.time_stamp + "errors.csv")
    
    with open(gold_file, "r", encoding='utf-8') as reader:
        gold_data = json.load(reader)["data"]
    
    gold = {}
    for entry in gold_data:
        for paragraph in entry['paragraphs']:
            for qa in paragraph['qas']:
                gold_key = qa['id']  # unique identifier for question
                gold_value = collections.OrderedDict({
                    'qid': gold_key, 
                    'title': entry['title'], 
                    'context': paragraph['context'],
                    'question': qa['question'],
                    'is_impossible': qa["is_impossible"],
                    'answers': qa['answers'],
                })
                gold[gold_key] = gold_value

    
    with open(input_prediction_file, "r") as reader:
        pred = json.load(reader)

    # load is_impossible predictions output from run_classifier.py; if qas id is in dict, then predicted is_impossible=True
    if args.ensemble:
        try:        
            with open('eval_predictions-pickle', "rb") as reader:
                predict_is_impossible = pickle.load(reader)
        except:
            predict_is_impossible = {}
    
    
    total_gold = len(gold)
    total_pred = len(pred)
    question_types = ['who', 'what', 'which', 'when', 'where', 'why', 'how']
    q_len = [collections.defaultdict(int) for _ in range(3)]  # Question length [no match, exact match, partial match]
    a_len = [collections.defaultdict(int) for _ in range(3)]  # Answer length [no match, exact match, partial match]
    c_len = [collections.defaultdict(int) for _ in range(3)]  # Context length [no match, exact match, partial match]
    q_type = [collections.defaultdict(int) for _ in range(3)] # Type of question
    
    no_match, partial_match = [], []
    avna = em_total = f1_total = total = 0

    for key, value in pred.items():
        total += 1
        ground_truths = gold[key]['answers']
        if args.ensemble and key in predict_is_impossible.keys():  # classifier overrides QA for is_impossible
            pred_answer = ''
        else:
            pred_answer = value
        em = metric_max_over_ground_truths(compute_em, pred_answer, ground_truths)
        f1 = metric_max_over_ground_truths(compute_f1, pred_answer, ground_truths)
        if args.version_2_with_negative:
            avna += compute_avna(pred_answer, ground_truths)
        
        if em == 1:
            match = 1  # exact match
        elif f1 == 0:
            match = 0  # no match
            example = gold[key]
            example['pred_answer'] = pred_answer
            example['f1'] = f1
            no_match.append(example)
        else:
            match = 2  # partial match
            example = gold[key]
            example['pred_answer'] = pred_answer
            example['f1'] = f1
            partial_match.append(example)
        
        em_total += em
        f1_total += f1

        a_len[match][len(pred_answer)] += 1 # increment answer length counter       
        q_len[match][len(gold[key]['question'])] += 1  # increment question length counter
        c_len[match][len(gold[key]['context'])] += 1  # increment context length counter
        
        if gold[key]['is_impossible']:
            q_type[match]['is_impossible'] += 1
        else:
            q_type[match]['is_possible'] += 1
            
        for item in question_types:
            if item == question_types[0]: found = False
            if gold[key]['question'].lower().find(item) >= 0:  # note could be more or less than 1-to-1 relationship
                q_type[match][item] += 1
                found = True
            if item == question_types[-1] and not found:
                for other_item in ['is', 'was', 'do', 'did', 'does', 'are', 'can', 'name', 'if', 'has', 'were', 'could']:
                    if gold[key]['question'].lower().split()[0] == other_item:
                        q_type[match]['other_' + other_item] += 1
                        found = True
                        break
                if not found:
                    print(gold[key]['question'])
                    q_type[match]['other'] += 1
            
    eval_dict = {'gold_file': gold_file,
                 'total_gold': total_gold,
                 'predict_file': input_prediction_file,
                 'total_pred': total_pred,
                 'EM': 100. * em_total / total,
                 'F1': 100. * f1_total / total}

    if args.version_2_with_negative:
        eval_dict['AvNA'] = 100. * avna / total

    # ----------------------------------------------------------
    # Save model errors and performance statistics
    # ----------------------------------------------------------

    with open(output_error_analysis_file + '.json', "w") as writer:
        for summary_data in [eval_dict, q_type, q_len, a_len]:
            writer.write("-"*80 + "\n")
            writer.write(json.dumps(summary_data, sort_keys=True, indent=4) + "\n")
        for example in no_match:
            writer.write("-"*80 + "\n")
            writer.write(json.dumps(example, indent=4) + "\n")
        for example in partial_match:
            writer.write("-"*80 + "\n")
            writer.write(json.dumps(example, indent=4) + "\n")
    
    # Setup columns and rows for DataFrames    
    match_types = ['no_match', 'exact_match', 'partial_match']
    question_possible = ['is_possible', 'is_impossible']
    question_types = []
    for i in range(len(match_types)):
        for key in q_type[i].keys():
            if key not in question_possible and key not in question_types:
                question_types.append(key)

    # Create empty DataFrames
    df_q_possible = pd.DataFrame(columns = match_types, index = question_possible)
    df_q_type = pd.DataFrame(columns = match_types, index = question_types)
    df_c_len = pd.DataFrame(columns = match_types, index = list(range(50 * args.max_seq_length)))
    df_q_len = pd.DataFrame(columns = match_types, index = list(range(50 * args.max_query_length)))
    df_a_len = pd.DataFrame(columns = match_types, index = list(range(50 * args.max_answer_length)))
    
    # Convert Dicts to DataFrame for easy ExcelWriter
    df_eval_dict = pd.DataFrame(list(eval_dict.values()), index= eval_dict.keys())

    for col_num, col_name in enumerate(match_types):
        for row in question_possible:
            df_q_possible.loc[row][col_name] = q_type[col_num][row]
    df_q_possible.fillna(value=0, inplace=True)
    df_q_possible['total'] = df_q_possible.no_match + df_q_possible.exact_match + df_q_possible.partial_match
    df_q_possible.sort_values(by= ['total'] , ascending=False, inplace=True)

    for col_num, col_name in enumerate(match_types):
        for row in question_types:
            df_q_type.loc[row][col_name] = q_type[col_num][row]
    df_q_type.fillna(value=0, inplace=True)
    df_q_type['total'] = df_q_type.no_match + df_q_type.exact_match + df_q_type.partial_match
    df_q_type.sort_values(by= ['total'] , ascending=False, inplace=True)
    
    for col_num, col_name in enumerate(match_types):
        for row, value in c_len[col_num].items():
            df_c_len.iloc[row][col_name] = value
    df_c_len.fillna(value=0, inplace=True)
    # df_c_len.drop_duplicates(keep= False, inplace= True)
    df_c_len['total'] = df_c_len.no_match + df_c_len.exact_match + df_c_len.partial_match
    df_c_len = df_c_len[df_c_len.total > 0]

    for col_num, col_name in enumerate(match_types):
        for row, value in q_len[col_num].items():
            df_q_len.iloc[row][col_name] = value
    df_q_len.fillna(value=0, inplace=True)
    # df_q_len.drop_duplicates(keep= False, inplace= True)
    df_q_len['total'] = df_q_len.no_match + df_q_len.exact_match + df_q_len.partial_match
    df_q_len = df_q_len[df_q_len.total > 0]

    for col_num, col_name in enumerate(match_types):
        for row, value in a_len[col_num].items():
            df_a_len.iloc[row][col_name] = value
    df_a_len.fillna(value=0, inplace=True)
    # df_a_len.drop_duplicates(keep= False, inplace= True)
    df_a_len['total'] = df_a_len.no_match + df_a_len.exact_match + df_a_len.partial_match
    df_a_len = df_a_len[df_a_len.total > 0]
    
    # Write DataFrames to Excel
    writer = pd.ExcelWriter(output_error_analysis_file + '.xlsx', engine='xlsxwriter',
                            datetime_format='yyyy-mm-dd', date_format='yyyy-mm-dd')
    workbook = writer.book
    col_format = workbook.add_format({'align': 'right',})
    ticker_idx = True
    startcol = 2
    startrow = 2
    df_eval_dict.to_excel(writer, sheet_name= 'errorAnalysis', startrow= startrow, startcol= startcol,
                     header= False, index= ticker_idx)
    startrow += len(df_eval_dict) + 2
    worksheet = writer.sheets['errorAnalysis']
    
    df_q_possible.to_excel(writer, sheet_name= 'errorAnalysis', startrow= startrow, startcol= startcol,
                     header= True, index= ticker_idx)
    worksheet.write(startrow - 1, startcol, "Answerability of Question")
    
    offset = startrow + len(df_q_possible) + 4
    df_q_type.to_excel(writer, sheet_name= 'errorAnalysis', startrow= offset, startcol= startcol,
                     header= True, index= ticker_idx)
    worksheet.write(offset - 1, startcol, "Type of Question")
   
    startcol = 10
    startrow = startrow
    df_c_len.to_excel(writer, sheet_name= 'errorAnalysis', startrow= startrow, startcol= startcol,
                     header= True, index= ticker_idx)
    worksheet.write(startrow - 1, startcol, "Length of Context")
    
    startcol = 17
    startrow = startrow
    df_q_len.to_excel(writer, sheet_name= 'errorAnalysis', startrow= startrow, startcol= startcol,
                     header= True, index= ticker_idx)
    worksheet.write(startrow - 1, startcol, "Length of Question")

    startcol = 24
    startrow = startrow
    df_a_len.to_excel(writer, sheet_name= 'errorAnalysis', startrow= startrow, startcol= startcol,
                     header= True, index= ticker_idx)
    worksheet.write(startrow - 1, startcol, "Length of Answer")
    
    
    # TODO: Write four new worksheet tabs for [summary, detail] examples of [no_match, partial_match]
    
    #worksheet = writer.sheets['errorAnalysis']
    #worksheet.set_column(0, ticker_col, 20, col_format)
    #worksheet.write(1, ticker_col - 1, ticker)
    writer.save()
    
        
    
    ## CSV for easy import into MSExcel
    ## TODO--Quick-and-dirty; change this to Pandas
    #csv.register_dialect('myDialect', delimiter = ',', quoting=csv.QUOTE_NONE)
    #with open(output_error_analysis_file +'.csv', 'w') as csv_fh:
        #fieldnames = ['key', 'value']
        ##csv_writer = csv.DictWriter(csv_fh, fieldnames=fieldnames, dialect ='myDialect')
        #csv_writer = csv.writer(csv_fh, dialect ='myDialect')
        #for summary_data in [eval_dict, q_type[0], q_type[1], q_type[2], q_len[0], q_len[1], q_len[2], a_len[0], a_len[1], a_len[2]]:
            ##csv_writer.writerow('eval_dict')
            ##csv_writer.writeheader()
            ##csv_writer.writerows(summary_data)
            #for k, v in summary_data.items():
                #csv_writer.writerow([k, v])
        
            

    return eval_dict

def read_squad_examples(input_file, is_training, version_2_with_negative, tiny_data=False, is_trivaqa=False):
    """Read a SQuAD json file into a list of SquadExample."""
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]
        
    def save_squad_examples(input_file, examples):
        with open(input_file.split('.json')[0] + '-pickled', "wb") as writer:
            pickle.dump(examples, writer)     

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    num_impossible = 0
    examples = []
    for entry in input_data:
        if tiny_data and len(examples) > TINY_DATA_SIZE:
            break
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None
                is_impossible = False
                if is_training:
                    if version_2_with_negative:
                        is_impossible = qa["is_impossible"]
                    if (len(qa["answers"]) != 1) and (not is_impossible):
                        raise ValueError(
                            "For training, each question should have exactly 1 answer.")
                    if not is_impossible:
                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"]
                        answer_offset = answer["answer_start"]
                        answer_length = len(orig_answer_text)
                        if (answer_offset + answer_length - 1) >= len(char_to_word_offset):
                            logger.warning("Answer beyond bounds of context string: '%s' vs. '%s'",
                                           orig_answer_text, answer_offset)
                            continue                            
                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[answer_offset + answer_length - 1]
                        # Only add answers where the text can be exactly recovered from the
                        # document. If this CAN'T happen it's likely due to weird Unicode
                        # stuff so we will just skip the example.
                        #
                        # Note that this means for training mode, every example is NOT
                        # guaranteed to be preserved.
                        actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                        cleaned_answer_text = " ".join(
                            whitespace_tokenize(orig_answer_text))
                        is_trivaqa = True  # debug
                        if is_trivaqa and actual_text.find(cleaned_answer_text) == -1:  # try all lower case if TriviaQA mismatch
                            actual_text = actual_text.lower()
                            cleaned_answer_text = cleaned_answer_text.lower()
                        if actual_text.find(cleaned_answer_text) == -1:
                            logger.warning("Could not find answer: {} vs. {}".format(
                                           actual_text, cleaned_answer_text))
                            continue
                    else:
                        start_position = -1
                        end_position = -1
                        orig_answer_text = ""
                        num_impossible += 1

                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible)
                examples.append(example)
                if tiny_data and len(examples) == TINY_DATA_SIZE:
                    save_squad_examples(input_file, examples)
                    return examples

    logger.info("***** Read Squad Examples *****")
    logger.info("  Num Squad examples = %d", len(examples))
    logger.info("  Num Squad examples with No Answer = %d", num_impossible)
    save_squad_examples(input_file, examples)
    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000

    features = []
    for (example_index, example) in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None
        if is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
        if is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example.orig_answer_text)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            start_position = None
            end_position = None
            if is_training and not example.is_impossible:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start and
                        tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                else:
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset
            if is_training and example.is_impossible:
                start_position = 0
                end_position = 0
            if example_index < TINY_DATA_SIZE:
                logger.info("*** Example ***")
                logger.info("unique_id: %s" % (unique_id))
                logger.info("example_index: %s" % (example_index))
                logger.info("doc_span_index: %s" % (doc_span_index))
                logger.info("tokens: %s" % " ".join(tokens))
                logger.info("token_to_orig_map: %s" % " ".join([
                    "%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
                logger.info("token_is_max_context: %s" % " ".join([
                    "%d:%s" % (x, y) for (x, y) in token_is_max_context.items()
                ]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info(
                    "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                if is_training and example.is_impossible:
                    logger.info("impossible example")
                if is_training and not example.is_impossible:
                    answer_text = " ".join(tokens[start_position:(end_position + 1)])
                    logger.info("start_position: %d" % (start_position))
                    logger.info("end_position: %d" % (end_position))
                    logger.info(
                        "answer: %s" % (answer_text))

            features.append(
                InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=example.is_impossible))
            unique_id += 1

    return features


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])


def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file, verbose_logging,
                      version_2_with_negative, null_score_diff_threshold, args=None):
    """Write final predictions to the json file and log-odds of null if needed."""
    logger.info("Writing predictions to: %s" % (output_prediction_file))
    logger.info("Writing nbest to: %s" % (output_nbest_file))
    
    # load is_impossible predictions output from run_classifier.py; if qas id is in dict, then predicted is_impossible=True
    if args.ensemble:
        try:        
            with open('eval_predictions-pickle', "rb") as reader:
                predict_is_impossible = pickle.load(reader)
        except:
            predict_is_impossible = {}
    

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min mull score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if version_2_with_negative:
                feature_null_score = result.start_logits[0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))
        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit))
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))
        # if we didn't include the empty option in the n-best, include it
        if version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(
                    _NbestPrediction(
                        text="",
                        start_logit=null_start_logit,
                        end_logit=null_end_logit))
                
            # In very rare edge cases we could only have single null prediction.
            # So we just create a nonce prediction in this case to avoid failure.
            if len(nbest)==1:
                nbest.insert(0,
                    _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))
                
        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if not version_2_with_negative:
            all_predictions[example.qas_id] = nbest_json[0]["text"]
        else:
            # predict "" iff the null score - the score of best non-null > threshold
            score_diff = score_null - best_non_null_entry.start_logit - (
                best_non_null_entry.end_logit)
            scores_diff_json[example.qas_id] = score_diff
            if score_diff > null_score_diff_threshold:
                all_predictions[example.qas_id] = ""
            else:
                if args.ensemble and example.qas_id in predict_is_impossible.keys():  # classifier overrides QA for is_impossible
                    all_predictions[example.qas_id] = ""
                else:
                    all_predictions[example.qas_id] = best_non_null_entry.text
                    all_nbest_json[example.qas_id] = nbest_json

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")
        
    # KML write prediction file in CSV format required by DFP Leaderboard p18 .pdf v2
    # fix output_prediction_file
    output_submission_file = output_prediction_file.split('.json')[0] + '_submission.csv' 
    logger.info("Writing predictions submission in DFP Leaderboard compliant format to: %s" % (output_submission_file))
    with open(output_submission_file, 'w', encoding='utf-8') as csv_fh:
        csv_writer = csv.writer(csv_fh, delimiter=',')
        csv_writer.writerow(['Id', 'Predicted'])
        for uuid in sorted(all_predictions):
            csv_writer.writerow([uuid, all_predictions[uuid]])
    

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    if version_2_with_negative:
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                        orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def load_split_data(args, data, split=0.9):
    """Split training data into train and validation"""
    
    if args.log_traindev_loss:
        to_train = int(len(data) * split)
        #train_data = data[:to_train]
        #val_data = data[to_train:]
        train_data = Subset(data, range(to_train))
        val_data = Subset(data, range(to_train, len(data)))
    else:
        train_data = data
        val_data = None
    
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
        if val_data is not None: val_sampler = RandomSampler(val_data)
            
    else:
        train_sampler = DistributedSampler(train_data)
        if val_data is not None: val_sampler = DistributedSampler(val_data)
        
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
    if val_data is not None:
        val_dataloader = DataLoader(val_data, sampler= val_sampler, batch_size=args.train_batch_size)
    else:
        val_dataloader = None
        
    return train_dataloader, val_dataloader


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")

    ## Other parameters
    parser.add_argument("--time_stamp", default=None, type=str, help="YYDDMM-HH_MM- to load a specific model file in output directory")  # KML
    parser.add_argument("--log_traindev_loss", action='store_true', help="Whether to use 10% of train data as dev set and log train/dev loss")  # KML
    parser.add_argument("--val_steps", default= 500, type=int, help="Number of training steps to take between validation measurements")  # KML
    parser.add_argument("--patience", default=5, type=int, help="Number of validations to wait without new best loss before aborting training")  # KML
    parser.add_argument("--tiny_data", action='store_true', help="Whether to use just 100 train/dev examples to debug code")  # KML
    parser.add_argument("--add_triviaqa_train", action='store_true', help="Whether to add TriviaQA examples to train. Postpend -triviaqa.json to --train_file")  # KML
    parser.add_argument("--ensemble", action='store_true', help="Whether to ensemble Classifier and QA models.")  # KML
    parser.add_argument("--load_eval_results_pickle", action='store_true', help="Whether to load previous eval predictions.")  # KML
    parser.add_argument("--train_file", default=None, type=str, help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--predict_file", default=None, type=str,
                        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_predict", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_error_analysis", action='store_true', help="Whether to run error analysis on the dev set vs predictions.")  # KML
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--predict_batch_size", default=8, type=int, help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
                             "of training.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--version_2_with_negative',
                        action='store_true',
                        help='If true, the SQuAD examples contain some that do not have an answer.')
    parser.add_argument('--null_score_diff_threshold',
                        type=float, default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.")
    args = parser.parse_args()

    # KML No longer necessary since addition of args.time_stamp to file naming convention
    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
    #    raise ValueError("Output directory () already exists and is not empty.")
    
    # KML create args.time_stamp for output file naming convention
    if not args.time_stamp:
        args.time_stamp = datetime.datetime.today().strftime("%y%m%d-%H_%M-")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # KML save runtime logs to disk
    output_log_file = os.path.join(args.output_dir, args.time_stamp + "log.log")
    logger.addHandler(logging.FileHandler(output_log_file, 'w', 'utf-8'))
    
    # KML Setup tensorboard logging
    if args.log_traindev_loss:
        tensorboard = Tensorboard('logs')

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_predict:
        if args.do_error_analysis:
            write_error_analysis(args)
            exit()
        else:
            raise ValueError("At least one of `do_train` or `do_predict` or 'do_error_analysis' must be True.")

    if args.do_train:
        if not args.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")
    if args.do_predict:
        if not args.predict_file:
            raise ValueError(
                "If `do_predict` is True, then `predict_file` must be specified.")
    
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = read_squad_examples(
            input_file=args.train_file, is_training=True, version_2_with_negative=args.version_2_with_negative,
            tiny_data=args.tiny_data)
        if args.add_triviaqa_train:
            train_examples_triviaqa = read_squad_examples(
                input_file=args.train_file.split('.json')[0] + '-triviaqa.json', is_training=True, version_2_with_negative=args.version_2_with_negative,
                tiny_data=args.tiny_data, is_trivaqa=True)
            train_examples += train_examples_triviaqa[0:-1]
        # Adjust if necessary for Train/Dev sub-split for loss tracking
        if args.log_traindev_loss:
            split_multiplier = 0.9
        else:
            split_multiplier = 1.0
        num_train_optimization_steps = int(
            (len(train_examples) * split_multiplier) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    print(PYTORCH_PRETRAINED_BERT_CACHE)
    model = BertForQuestionAnswering.from_pretrained(args.bert_model,
                cache_dir=os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank)))

    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    global_step = 0
    if args.do_train:
        cached_train_features_file = args.train_file+'_{0}_{1}_{2}_{3}_{4}_{5}'.format(
            list(filter(None, args.bert_model.split('/'))).pop(), str(args.add_triviaqa_train), str(args.tiny_data), str(args.max_seq_length), str(args.doc_stride), str(args.max_query_length))
        train_features = None
        try:
            if False:  # args.tiny_data:
                raise Exception('Ignoring cached features file because --tiny_data flag is set')
            else:                
                with open(cached_train_features_file, "rb") as reader:
                    train_features = pickle.load(reader)
        except:
            train_features = convert_examples_to_features(
                examples=train_examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length,
                is_training=True)
            if (args.local_rank == -1 or torch.distributed.get_rank() == 0):  # not args.tiny_data and
                logger.info("  Saving train features into cached file %s", cached_train_features_file)
                with open(cached_train_features_file, "wb") as writer:
                    pickle.dump(train_features, writer)

        #if args.log_traindev_loss:
            #train_end_index = int(len(train_examples) * split_multiplier)
            #train_examples, dev_examples = train_examples[0:train_end_index], train_examples[train_end_index+1, :]

        logger.info("***** Running training *****")
        logger.info("  Num orig examples = %d", len(train_examples))
        logger.info("  Num split examples = %d", len(train_features))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                   all_start_positions, all_end_positions)
        
        train_dataloader, val_dataloader = load_split_data(args, train_data)
        
        #if args.local_rank == -1:
            #train_sampler = RandomSampler(train_data)
        #else:
            #train_sampler = DistributedSampler(train_data)
        #train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        running_loss = 0
        all_steps = -1
        best_val, best_step = 10000, 0
        patience = args.patience
        loss_history = []
        for e in trange(int(args.num_train_epochs), desc="Epoch"):
            if e == 0:
                # Freeze all pretrained layers but not the QA layers for the first epoch for better performance
                logger.info("  *** Freezing Pretrained BERT Layers")
                no_freeze = ['module.qa_intermediate1.weight', 'module.qa_intermediate1.bias', 'module.qa_intermediate2.weight', \
                             'module.qa_intermediate2.bias', 'module.qa_outputs.weight', 'module.qa_outputs.bias']
                for name, param in model.named_parameters():
                    if param.requires_grad == False:
                        logger.info("  --> requires_grad already equal to False for: {}".format(name))
                    if name not in no_freeze:
                        param.requires_grad == False
                        logger.info("  --> set requires_grad equal to False for: {}".format(name))
            else:        
                if patience == 0: break
                # Unfreeze all layers for subsequent epochs
                logger.info("  *** Unfreezing Pretrained BERT Layers")
                #no_freeze = ['qa_intermediate1.weight', 'qa_intermediate1.bias', 'qa_intermediate2.weight', \
                #             'qa_intermediate2.bias', 'qa_outputs.weight', 'qa_outputs.bias']
                for name, param in model.named_parameters():
                    param.requires_grad = True
                
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                if patience == 0: break
                if n_gpu == 1:
                    batch = tuple(t.to(device) for t in batch) # multi-gpu does scattering it-self
                input_ids, input_mask, segment_ids, start_positions, end_positions = batch
                loss = model(input_ids, segment_ids, input_mask, start_positions, end_positions)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used and handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(global_step/num_train_optimization_steps, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                
                all_steps += 1
                running_loss += loss.item()
                    
                if all_steps % args.val_steps == 0 and (args.log_traindev_loss):
                    val_loss = 0
                    model.eval()
                    with torch.no_grad():
                        for batch in val_dataloader:
                            if n_gpu == 1:
                                batch = tuple(t.to(device) for t in batch) # multi-gpu does scattering it-self
                            input_ids, input_mask, segment_ids, start_positions, end_positions = batch
                            batch_loss = model(input_ids, segment_ids, input_mask, start_positions, end_positions)
                            if n_gpu > 1:
                                batch_loss = batch_loss.mean() # mean() to average on multi-gpu.
                            if args.gradient_accumulation_steps > 1:
                                batch_loss = batch_loss / args.gradient_accumulation_steps
                            val_loss += batch_loss.item()
                    tensorboard.log_scalar('train loss', running_loss / 1, all_steps)  # len(train_dataloader)
                    tensorboard.log_scalar('val loss', val_loss / len(val_dataloader), all_steps)
                    loss_history.append((all_steps, running_loss, val_loss))
                    
                    if all_steps >= args.val_steps * 1:
                        if (val_loss / len(val_dataloader)) < best_val:
                            patience = args.patience
                            best_val = (val_loss / len(val_dataloader))
                            best_step = all_steps
                            logger.info("*** Saving best validation model at step {} with loss of {}".format(best_step, val_loss))
                            #model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                            #torch.save(model_to_save.state_dict(), 'best_val_model')
                            torch.save(model, 'best_val_model')
                        else:
                            patience -= 1
                            
                    running_loss = 0
                    model.train()
        
        # Load the best saved validation checkpoint model, if it exists            
        if args.log_traindev_loss:
            try:
                #model.load_state_dict(torch.load('best_val_model'))
                model = torch.load('best_val_model')
                logger.info("*** Loading best validation model")
            except:
                logger.info("*** WARNING: could not load best saved validation model")

    if args.do_train:
        # Save a trained model and the associated configuration
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, args.time_stamp + WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(args.output_dir, args.time_stamp + CONFIG_NAME)
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())

        # Load a trained model and config that you have fine-tuned
        config = BertConfig(output_config_file)
        model = BertForQuestionAnswering(config)
        model.load_state_dict(torch.load(output_model_file))
    else:
        #try:
            # Load a previously saved tuned model specified by --time_stamp
            # note: cannot reload a model .bin file on local CPU machine that was trained on remote GPU machine
        output_model_file = os.path.join(args.output_dir, args.time_stamp + WEIGHTS_NAME)
        output_config_file = os.path.join(args.output_dir, args.time_stamp + CONFIG_NAME)            
        config = BertConfig(output_config_file)
        logger.info("**Success: loaded {}".format(output_config_file))
        model = BertForQuestionAnswering(config)
        logger.info("**Success: initialized empty BertQA model")
        model.load_state_dict(torch.load(output_model_file, map_location=device))  # KML this should map GPU to CPU models
        # model = torch.load("190317-20_37-best_val_model")
        logger.info("**Success: loaded {}".format(output_model_file))
        #except:
            #model = BertForQuestionAnswering.from_pretrained(args.bert_model)

    model.to(device)

    if args.do_predict and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        if not args.load_eval_results_pickle:
                
            eval_examples = read_squad_examples(
                input_file=args.predict_file, is_training=False, version_2_with_negative=args.version_2_with_negative, 
                tiny_data=args.tiny_data)
            eval_features = convert_examples_to_features(
                examples=eval_examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length,
                is_training=False)
    
            # dump a copy of eval_features for run_classifier.py
            cached_eval_features_file = args.predict_file+'_{0}_{1}_{2}_{3}_{4}_{5}'.format(
                list(filter(None, args.bert_model.split('/'))).pop(), str(False), str(args.tiny_data), str(args.max_seq_length), str(args.doc_stride), str(args.max_query_length)) 
            logger.info("  Saving eval features into cached file %s", cached_eval_features_file)
            with open(cached_eval_features_file, "wb") as writer:
                pickle.dump(eval_features, writer)
    
            logger.info("***** Running predictions *****")
            logger.info("  Num orig examples = %d", len(eval_examples))
            logger.info("  Num split examples = %d", len(eval_features))
            logger.info("  Batch size = %d", args.predict_batch_size)
    
            all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
            all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
            # Run prediction for full data
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size)
    
            model.eval()
            all_results = []
            logger.info("Start evaluating")
            for input_ids, input_mask, segment_ids, example_indices in tqdm(eval_dataloader, desc="Evaluating"):
                if len(all_results) % 1000 == 0:
                    logger.info("Processing example: %d" % (len(all_results)))
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                with torch.no_grad():
                    batch_start_logits, batch_end_logits = model(input_ids, segment_ids, input_mask)
                for i, example_index in enumerate(example_indices):
                    start_logits = batch_start_logits[i].detach().cpu().tolist()
                    end_logits = batch_end_logits[i].detach().cpu().tolist()
                    eval_feature = eval_features[example_index.item()]
                    unique_id = int(eval_feature.unique_id)
                    all_results.append(RawResult(unique_id=unique_id,
                                                 start_logits=start_logits,
                                                 end_logits=end_logits))
            
            # debug flag -- set to False 
            if True:
                logger.info("  Saving pickle dump of eval_examples, eval_features, all_results for debug analysis")
                #cached_eval_results_file = args.predict_file+'_{0}_{1}'.format(
                #list(filter(None, args.bert_model.split('/'))).pop(), 'pickle_eval_results' )
                cached_eval_results_file = 'all_results-pickle'
                with open(cached_eval_results_file, "wb") as writer:
                    pickle.dump([eval_examples, eval_features, all_results], writer)
        
        else:
            cached_eval_results_file = 'all_results-pickle'
            with open(cached_eval_results_file, "rb") as reader:
                [eval_examples, eval_features, all_results] = pickle.load(reader)

        output_prediction_file = os.path.join(args.output_dir, args.time_stamp + "predictions.json")
        output_nbest_file = os.path.join(args.output_dir, args.time_stamp + "nbest_predictions.json")
        output_null_log_odds_file = os.path.join(args.output_dir, args.time_stamp + "null_odds.json")
            
        write_predictions(eval_examples, eval_features, all_results,
                          args.n_best_size, args.max_answer_length,
                          args.do_lower_case, output_prediction_file,
                          output_nbest_file, output_null_log_odds_file, args.verbose_logging,
                          args.version_2_with_negative, args.null_score_diff_threshold, args)


        if args.do_error_analysis:
            write_error_analysis(args)
        
        try:
            tensorboard
        except:
            pass
        else:
            tensorboard.close()
        
if __name__ == "__main__":
    main()
