#!/bin/bash

cat hepg2_expressed_list.txt | sed 's/[][]//g' | sed 's/ //g' > hepg2_expressed_list_v2.txt
