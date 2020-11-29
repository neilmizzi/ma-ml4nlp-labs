import sys
import pandas as pd
# see tips & tricks on using defaultdict (remove when you do not use it)
from collections import defaultdict, Counter
import numpy as np



def obtain_counts(goldannotations, machineannotations):
    '''
    This function compares the gold annotations to machine output
    
    :param goldannotations: the gold annotations
    :param machineannotations: the output annotations of the system in question
    :type goldannotations: the type of the object created in extract_annotations
    :type machineannotations: the type of the object created in extract_annotations
    
    :returns: a countainer providing the counts for each predicted and gold class pair
    '''
    
    # TIP on how to get the counts for each class
    # https://stackoverflow.com/questions/49393683/how-to-count-items-in-a-nested-dictionary, last accessed 22.10.2020
    evaluation_counts = defaultdict(Counter)
    assert len(goldannotations) == len(machineannotations)
    for i in range(len(goldannotations)):
        evaluation_counts[goldannotations[i]][machineannotations[i]] += 1
    return evaluation_counts
    
def calculate_precision_recall_fscore(evaluation_counts):
    '''
    Calculate precision recall and fscore for each class and return them in a dictionary
    
    :param evaluation_counts: a container from which you can obtain the true positives, false positives and false negatives for each class
    :type evaluation_counts: type of object returned by obtain_counts
    
    :returns the precision, recall and f-score of each class in a container
    '''
    
    # TIP: you may want to write a separate function that provides an overview of true positives, false positives and false negatives
    #      for each class based on the outcome of obtain counts
    # YOUR CODE HERE (and remove statement below)
    value_dict = defaultdict(Counter)

    macro_list_prec = []
    macro_list_rec = []
    macro_list_fscore = []
    macro_count = 0
    
    for label in evaluation_counts.keys():
        # Calculation of TP, FP & FN
        TP = evaluation_counts[label][label]
        FP = 0
        FN = 0
        for label_ in evaluation_counts.keys():
            if label_ != label:
                FP += evaluation_counts[label_][label]
                FN += evaluation_counts[label][label_]

        # Precision, Recall & F-Score for class 'label'
        try:
            precis = TP / (TP + FP)
        except ZeroDivisionError:
            precis = 0
        try:
            recall = TP / (TP + FN)
        except ZeroDivisionError:
            recall = 0
        try:
            FScore = (2 * precis * recall) / (precis + recall)
        except ZeroDivisionError:
            FScore = 0
        
        # Assign Precision, Recall, & F-Score according to set structure
        value_dict[label]['precision'] = precis
        value_dict[label]['recall'] = recall
        value_dict[label]['f-score'] = FScore

        macro_list_fscore.append(FScore)
        macro_list_prec.append(precis)
        macro_list_rec.append(recall)
        macro_count += 1
    
    value_dict['macro']['precision'] = sum(macro_list_prec) / float(macro_count)
    value_dict['macro']['recall'] = sum(macro_list_rec) / float(macro_count)
    value_dict['macro']['f-score'] = sum(macro_list_fscore) / float(macro_count)

    return value_dict
            
def provide_confusion_matrix(evaluation_counts):
    '''
    Read in the evaluation counts and provide a confusion matrix for each class
    
    :param evaluation_counts: a container from which you can obtain the true positives, false positives and false negatives for each class
    :type evaluation_counts: type of object returned by obtain_counts
    
    :prints out a confusion matrix
    '''
    
    # TIP: provide_output_tables does something similar, but those tables are assuming one additional nested layer
    #      your solution can thus be a simpler version of the one provided in provide_output_tables below
    
    # YOUR CODE HERE (and remove statement below)
    res = pd.DataFrame.from_dict(evaluation_counts).fillna(0)

    print(res)
    print(res.to_latex())

def carry_out_evaluation(gold_annotations, systemfile, systemcolumn, delimiter='\t'):
    '''
    Carries out the evaluation process (from input file to calculating relevant scores)
    
    :param gold_annotations: list of gold annotations
    :param systemfile: path to file with system output
    :param systemcolumn: indication of column with relevant information
    :param delimiter: specification of formatting of file (default delimiter set to '\t')
    
    returns evaluation information for this specific system
    '''
    system_annotations = extract_annotations(systemfile, systemcolumn, delimiter)
    evaluation_counts = obtain_counts(gold_annotations, system_annotations)
    provide_confusion_matrix(evaluation_counts)
    evaluation_outcome = calculate_precision_recall_fscore(evaluation_counts)
    
    return evaluation_outcome

def provide_output_tables(evaluations):
    '''
    Create tables based on the evaluation of various systems
    
    :param evaluations: the outcome of evaluating one or more systems
    '''
    #https:stackoverflow.com/questions/13575090/construct-pandas-dataframe-from-items-in-nested-dictionary
    evaluations_pddf = pd.DataFrame.from_dict({(i,j): evaluations[i][j]
                                              for i in evaluations.keys()
                                              for j in evaluations[i].keys()},
                                             orient='index')
    print(evaluations_pddf)
    print(evaluations_pddf.to_latex())

def run_evaluations(goldfile, goldcolumn, systems):
    '''
    Carry out standard evaluation for one or more system outputs
    
    :param goldfile: path to file with goldstandard
    :param goldcolumn: indicator of column in gold file where gold labels can be found
    :param systems: required information to find and process system output
    :type goldfile: string
    :type goldcolumn: integer
    :type systems: list (providing file name, information on tab with system output and system name for each element)
    
    :returns the evaluations for all systems
    '''
    evaluations = {}
    #not specifying delimiters here, since it corresponds to the default ('\t')
    gold_annotations = extract_annotations(goldfile, goldcolumn)
    for system in systems:
        sys_evaluation = carry_out_evaluation(gold_annotations, system[0], system[1])
        evaluations[system[2]] = sys_evaluation
    return evaluations

def get_macro_score(list_1, list_2):
    evaluation_counts = obtain_counts(list_1, list_2)
    value_dict = defaultdict(Counter)
    macro_list_prec = []
    macro_list_rec = []
    macro_list_fscore = []
    macro_count = 0
    
    for label in evaluation_counts.keys():
        # Calculation of TP, FP & FN
        TP = evaluation_counts[label][label]
        FP = 0
        FN = 0
        for label_ in evaluation_counts.keys():
            if label_ != label:
                FP += evaluation_counts[label_][label]
                FN += evaluation_counts[label][label_]

        # Precision, Recall & F-Score for class 'label'
        try:
            precis = TP / (TP + FP)
        except ZeroDivisionError:
            precis = 0
        try:
            recall = TP / (TP + FN)
        except ZeroDivisionError:
            recall = 0
        try:
            FScore = (2 * precis * recall) / (precis + recall)
        except ZeroDivisionError:
            FScore = 0
        
        # Assign Precision, Recall, & F-Score according to set structure
        value_dict[label]['precision'] = precis
        value_dict[label]['recall'] = recall
        value_dict[label]['f-score'] = FScore

        macro_list_fscore.append(FScore)
        macro_list_prec.append(precis)
        macro_list_rec.append(recall)
        macro_count += 1

    prec = sum(macro_list_prec) / float(macro_count)
    rec = sum(macro_list_rec) / float(macro_count)
    f = sum(macro_list_fscore) / float(macro_count)
    
    return prec, rec, f


def compare_outcome(label_set_1, label_set_2):

    eval_counts = obtain_counts(label_set_1, label_set_2)

    values = calculate_precision_recall_fscore(eval_counts)
    values = pd.DataFrame.from_dict(values).fillna(0)
    print(values.to_latex())

    eval_counts = pd.DataFrame.from_dict(eval_counts).fillna(0)
    print(eval_counts.to_latex())
