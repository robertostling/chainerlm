#!/bin/sh

SRC=english.train
CORENLP=/usr/local/stanford-corenlp-full-2018-01-31

cut -f 1 "$SRC" >"$SRC".lang
cut -f 2 "$SRC" | split --suffix-length=2 --lines=10000 - "$SRC"_

for PART in "$SRC"_*; do
    echo $PART
    sem -j10 java -cp $CORENLP/'\*' -Xmx6g \
        edu.stanford.nlp.pipeline.StanfordCoreNLP \
        -props parser.props -file $PART
done
sem --wait

cat "$SRC"_*.xml | grep '<parse>' | \
    sed 's/ *<parse>//' | sed 's/ *<\/parse>//' \
    >"$SRC".parse

