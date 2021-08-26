#!/bin/bash

# ./make_html.sh < ./liver_enhancer/html_input_evalue_10_liver_enhancer.txt > ./liver_enhancer/evalue_10.html
# ./make_html.sh < ./silencer/html_input_evalue_10_silencer.txt > ./silencer/evalue_10.html

# The script requires a space-delimited data file to parse into an html table.
# It does not automatically create a header row.
width="150\%"
height="150\%"

echo "<table border=\"1\">"

echo "<tr>"
echo \<td\>"Filter Rank"\<\/td\>
echo \<td\>"Filter Number"\<\/td\>
echo \<td\>"fwd_logo"\<\/td\>
echo \<td\>"rc_logo"\<\/td\>
echo \<td\>"Tomtom_match"\<\/td\>
echo \<td\>"Dense_weight"\<\/td\>
echo "</tr>"

while read line; do
    echo "<tr>"

    fwd_logo=$(echo $line | cut -f1 -d' ')
    rc_logo=$(echo $line | cut -f2 -d' ')
    tomtom=$(echo $line | cut -f3 -d' ')
    filter_rank=$(echo $line | cut -f4 -d' ')
    filter_num=$(echo $line | cut -f5 -d' ')
    dense_weight=$(echo $line | cut -f6 -d' ')

    echo \<td\>$filter_rank\<\/td\>
    echo \<td\>$filter_num\<\/td\>
    echo "<td><img src=\"${fwd_logo}\" width=\"${width}\" height=\"${height}\"></td>"
    echo "<td><img src=\"${rc_logo}\" width=\"${width}\" height=\"${height}\"></td>"
    echo \<td\>$tomtom\<\/td\>
    echo \<td\>$dense_weight\<\/td\>

    echo "</tr>"
done
echo "</table>"

## Header
# filter fwd rc tomtom
