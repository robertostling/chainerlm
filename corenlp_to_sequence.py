# Script to convert CoreNLP output, with the following properties file:
#
#   annotators = tokenize, ssplit, pos, lemma, parse
#   outputExtension = .xml
#   outputFormat = xml
#   tokenize.whitespace = true
#   ssplit.eolonly = true
#   parse.model = edu/stanford/nlp/models/srparser/englishSR.beam.ser.gz
#
# Output is one sentence per line of partly delexicalized phrase structure
# trees, bracketed.
#
# NOTE: input should be preprocessed as follows before passed to this script:
#   grep '<parse>' file.xml | sed 's/ *<parse>//' | sed 's/ *<\/parse>//'
#
# NOTE: invalid trees will cause blank output lines, in practice this can
#       happen with weird space characters

import sys
import re

#RE_PAREN = re.compile(r'([) ]|(?:\(\))|(?:\([^\s)]+))')
RE_PAREN = re.compile(r'(\s|\)|\([^\s)]+)')
        
POS_DELEXICALIZE = set('''
    CD FW JJ JJR JJS NN NNS NNP NNPS RB RBR RBS VB VBD VBG VBN VBP VBZ
    '''.split())

LANG_CODES = set(
    'CS DA DE EL EN ES ET FI FR HU IT LT LV MT NL PL PT RO SK SL SV'.split())

def emit(tree):
    tag = tree[0]
    children = tree[1:]
    if tag in POS_DELEXICALIZE:
        yield tag
    elif len(children) == 1 and isinstance(children[0], str):
        yield children[0]
    else:
        yield '('+tag
        for child in children:
            if isinstance(child, str):
                yield child
            else:
                for token in emit(child):
                    yield token
        yield tag+')'

def process(line):
    stack = [[]]
    tokens = [s for s in RE_PAREN.split(line) if s.strip()]
    for token in tokens:
        if token in '()' and len(stack[-1]) == 1:
            # NOTE: special case for literal parentheses
            stack[-1].append(token)
        elif token[0] == '(':
            ptype = token[1:]
            stack.append([ptype])
        elif token == ')':
            tos = stack.pop()
            stack[-1].append(tos)
        else:
            if token in LANG_CODES:
                # just in case any mis-tagged language codes got through...
                token = 'XXX'
            token = token.lower()
            stack[-1].append(token)

    # NOTE: weird spaces in the data can cause this...
    if len(stack) != 1: return None
    #assert len(stack) == 1, stack
    #assert len(stack[0]) == 1
    return list(emit(stack[0][0]))


def main():
    for line in sys.stdin:
        tokens = process(line.strip())
        if tokens is None:
            print()
        else:
            print(' '.join(process(line.strip())))


if __name__ == '__main__':
    main()

