#!/bin/sh

# nli-shared-task-2017/data/essays/train/parsed$ cat *.xml | grep '<parse>' | sed 's/ *<parse>//' | sed 's/ *<\/parse>//' >nli-train.parse
# nli-shared-task-2017/data/essays/train/parsed$ for xml in *.xml; do grep -c '<parse>' $xml; done >nli.train.sentsperfile
# nli-shared-task-2017/data/labels/train$ tail -n +2 labels.train.csv | cut -d ',' -f 4 >nli.train.langperfile

PREFIX=/wrk/rostling/DONOTREMOVE/acl18/nli.train
PARSEFILE="$PREFIX".parse
TEXTFILE="$PREFIX".tokens
LANGFILE="$PREFIX".lang
COMBINED="$PREFIX".cfg
VOCABULARY="$PREFIX".vocab

python3 `dirname $0`/cleanup_nli_makelang.py \
    "$PREFIX".langperfile "$PREFIX".sentsperfile >"$LANGFILE"

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
        --tokenized --min-frequency 2 --exclude=de,le,wo,à,na,la,bam
fi

