#!/bin/bash

if [[ ! -f $1 ]] || [[ $# -ne 1 ]]; then
    echo "---style transfer script to csv---\n"
    echo "usage:\n\t$0 path-to-log.txt\n"
    echo "does the following:"
    echo "lines like:
        0   # 1.764 s inf 1.322e+10 || 4.36e8; 1.98e9; 4.59e9; 1.35e10; 2.53e9; 5.5e8; 
        20  # 0.728 s 1.384 9.550e+09 || 4.33e8; 5.32e8; 3.37e9; 6.42e9; 8.31e8; 8.5e8;"
    echo "will be converted as:
        2.70e8;9.75e8;2.10e9;4.62e9;3.50e9;1.66e10;1.100e10;8.50e10;
        1.13e9;2.34e9;4.48e9;1.36e10;6.8e8;4.11e9;5.72e9;3.15e10;\n"
    echo "out file will be path-to-log.csv"
    exit 1
fi

csv=$(echo $1 | sed 's#.txt$##').csv
cat $1 | grep '||' | sed -e 's#^.*||##' | tr -d ' ' | sed 's#;$##' > $csv

