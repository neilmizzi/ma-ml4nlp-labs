# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Preprocessing Files
# 
# Different sources and tools may make use of different formats to represent information and the output of various tools may not directly correspond. In this course, we will mainly (or even exclusively) work with the conll format. Even within this format, there may be differences in tokenization, class labels used or in the number of columns provided in the output. Depending on what the difference is exactly, you may want to adapt input files or build scripts that can deal with such differences during the process.
# In this case, we are preparing files that present output of two different tools for evaluation, where the exact annotation scheme differs. We set this up so you can first convert the files, so that they match and then can run evaluation (covered in a different notebook). Originally, both systems had a different tokenization and they both differed from the tokenization used in training and evaluation data. The steps of making sure that the tokens align have already been taken. We left some of the basic functions used as part of this process (e.g. the verification whether tokens align) as an example.

# %%
import csv
# csv is a useful package to deal with comma or tab separated values (such as conll). 
# It does not have quite the same functionality as pandas, but is easier to work with
import collections


# %%
def matching_tokens(conll1, conll2):
    '''
    Check whether the tokens of two conll files are aligned
    
    :param conll1: tokens (or full annotations) from the first conll file
    :param conll2: tokens (or full annotations) from the first conll file
    
    :returns boolean indicating whether tokens match or not
    '''
    for row in conll1:
        row2 = next(conll2)
        if row[0] != row2[0]:
            return False
    
    return True


# %%
def read_in_conll_file(conll_file, delimiter='\t'):
    '''
    Read in conll file and return structured object
    
    :param conll_file: path to conll_file
    :param delimiter: specifies how columns are separated. Tabs are standard in conll
    
    :returns structured representation of information included in conll file
    '''
    my_conll = open(conll_file, 'r')
    conll_as_csvreader = csv.reader(my_conll, delimiter=delimiter)
    return conll_as_csvreader


# %%
def alignment_okay(conll1, conll2):
    '''
    Read in two conll files and see if their tokens align
    '''
    my_first_conll = read_in_conll_file(conll1)
    my_second_conll = read_in_conll_file(conll2)
    
    return matching_tokens(my_first_conll, my_second_conll)
    
    


# %%
def get_predefined_conversions(conversion_file):
    '''
    Read in file with predefined conversions and return structured object that maps old annotation to new annotation
    
    :param conversion_file: path to conversion file
    
    :returns object that maps old annotations to new ones
    '''
    conversion_dict = {}
    my_conversions = open(conversion_file, 'r')
    conversion_reader = csv.reader(my_conversions, delimiter='\t')
    for row in conversion_reader:
        conversion_dict[row[0]] = row[1]
    return conversion_dict


# %%
def create_converted_output(conll_object, annotation_identifier, conversions, outputfilename):
    '''
    Check which annotations need to be converted for the output to match and convert them
    
    :param conll_object: structured object with conll annotations
    :param annotation_identifier: indicator of how to find the annotations in the object (e.g. key of dictionary, index of list)
    :param conversions: pointer to the conversions that apply. This can be external (e.g. a local file with conversions) or internal (e.g. prestructured dictionary). In case of an internal object, you probably want to add a function that creates this from a local file.
    
    '''
    with open(outputfilename, 'w') as outputcsv:
        csvwriter = csv.writer(outputcsv, delimiter='\t')
        for row in conll_object:
            annotation = row[annotation_identifier]
            if annotation in conversions:
                row[annotation_identifier] = conversions.get(annotation)
            csvwriter.writerow(row)


# %%
def write_out(conll_object, outputfilename):
    '''
    Write out (updated) conll object to an output file
    
    :param conll_object: the (updated) conll object
    :param outputfilename: path to the output file
    '''
    with open(outputfilename, 'w') as outputcsv:
        csvwriter = csv.writer(outputcsv, delimiter='\t')
        for row in conll_object:
            print('row')
            csvwriter.writerow(row)


# %%
def preprocess_files(conll1, conll2, column_identifiers, conversions):
    '''
    Guides the full process of preprocessing files and outputs the modified files.
    
    :param conll1: path to the first conll input file
    :param conll2: path to the second conll input file
    :param column_identifiers: object providing the identifiers for target column
    :param conversions: path to a file that defines conversions
    '''
    if alignment_okay(conll1, conll2):
        conversions = get_predefined_conversions(conversions)
        my_first_conll = read_in_conll_file(conll1)
        my_second_conll = read_in_conll_file(conll2)
        create_converted_output(my_first_conll, column_identifiers[0], conversions, conll1.replace('.conll','-preprocessed.conll'))
        create_converted_output(my_second_conll, column_identifiers[1], conversions, conll2.replace('.conll','-preprocessed.conll'))
        #converted_conll1 = convert_annotations(my_first_conll, column_identifiers[0], conversions)
        #converted_conll2 = convert_annotations(my_second_conll, column_identifiers[1], conversions)
        #write_out(converted_conll1, conll1.replace('.conll','-preprocessed.conll'))
        #write_out(converted_conll2, conll2.replace('.conll','-preprocessed.conll'))
    else:
        print(conll1, conll2, 'do not align')


# %%
preprocess_files('./data/spacy_out_matched_tokens.conll','./data/gold_stripped-preprocessed.conll', [2,3],'./code/settings/conversions.tsv')

preprocess_files('./data/stanford_out_matched_tokens.conll','./data/gold_stripped-preprocessed.conll', [3,3],'./code/settings/conversions.tsv')


# %%



