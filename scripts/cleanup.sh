#!/bin/sh

PREFIX=/wrk/rostling/DONOTREMOVE/acl18/english-beam.train
PARSEFILE="$PREFIX".parse
TEXTFILE="$PREFIX".tokens
LANGFILE="$PREFIX".lang
COMBINED="$PREFIX".cfg
VOCABULARY="$PREFIX".vocab

if [ ! -e $TEXTFILE ]; then
    echo "Converting parse trees to sequences..."
    cat $PARSEFILE | python3 corenlp_to_sequence.py >$TEXTFILE
fi

echo "Filtering out bad sentences and adding language tags..."
paste $LANGFILE $TEXTFILE | grep -v -P '\t$' >$COMBINED

if [ ! -e $VOCABULARY ]; then
    echo "Generating vocabulary..."
    python3 make_vocabulary_multi.py \
        --corpus $COMBINED --vocabulary $VOCABULARY \
        --tokenized --min-frequency 10 --exclude=de,le,wo,à,na,la,bam
fi

