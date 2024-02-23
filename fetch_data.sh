#!/bin/bash
set -ex

mkdir -p data_sources
if [ ! -f "./data_sources/drugs.csv" ]; then
  wget https://seer.cancer.gov/seertools/seerrx/download-rx/?type=drug -O \
       ./data_sources/drugs.csv
  python -m dataset_tools.chemo_drugs
fi
if [ ! -f "./data_sources/mtsamples2.csv" ]; then
  wget "https://raw.githubusercontent.com/mcoplan11/service_denial/master/dataset/mtsamples%202.csv" -O \
       ./data_sources/mtsamples2.csv || echo "huh no luck"
fi
if [ ! -f "./data_sources/wpath_soc7.pdf"]; then
  wget https://www.wpath.org/media/cms/Documents/SOC%20v7/SOC%20V7_English2012.pdf?_t=1613669341 -O \
       ./data_sources/wpath_soc7.pdf
fi
if [ ! -f "./data_sources/aca.pdf" ]; then
  wget http://housedocs.house.gov/energycommerce/ppacacon.pdf -O ./data_sources/aca.pdf
fi
# TBD: Should we include this?
if [ ! -f "./data_sources/ic10k.csv"] ; then
  wget https://drive.google.com/u/0/uc?id=1ZKbqgYqWc7DJHs3N9TQYQVPdDQmZaClA&export=download -O \
       ./data_sources/ic10k.csv
fi
if [ ! -f "./data_sources/wpath_soc8.pdf"]; then
  wget https://www.tandfonline.com/doi/pdf/10.1080/26895269.2022.2100644 -O \
       ./data_sources/wpath_soc8.pdf
fi
if [ ! -f "./data_sources/hiv_prep_soc.pdf"]; then
  wget https://www.cdc.gov/hiv/pdf/risk/prep/cdc-hiv-prep-guidelines-2021.pdf -O \
       ./data_sources/hiv_prep_soc.pdf
fi
if [ ! -f "./data_sources/erisa.pdf" ]; then
  wget https://www.govinfo.gov/content/pkg/COMPS-896/pdf/COMPS-896.pdf -O \
       ./data_sources/erisa.pdf
fi
if [! -f "./data_sources/ppacacon.pdf" ]; then
  wget http://housedocs.house.gov/energycommerce/ppacacon.pdf -O \
       ./data_sources/ppacacon.pdf
fi
if [ ! -f "./data_sources/ca-independent-medical-review-imr-determinations-trends.csv" ]; then
  # From https://data.chhs.ca.gov/dataset/independent-medical-review-imr-determinations-trend/resource/3340c5d7-4054-4d03-90e0-5f44290ed095
  # From https://data.chhs.ca.gov/dataset/independent-medical-review-imr-determinations-trend
  wget https://data.chhs.ca.gov/dataset/b79b3447-4c10-4ae6-84e2-1076f83bb24e/resource/3340c5d7-4054-4d03-90e0-5f44290ed095/download/independent-medical-review-imr-determinations-trends.csv -O \
       ./data_sources/ca-independent-medical-review-imr-determinations-trends.csv
  iconv -c -t utf-8 ./data_sources/ca-independent-medical-review-imr-determinations-trends.csv  > ./data_sources/ca-independent-medical-review-imr-determinations-trends-utf8.csv
fi
if [ ! -d "./data_sources/pubmed" ]; then
  mkdir -p ./data_sources/pubmed
  wget --recursive -e robots=off --no-parent https://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/ -P ./data_sources/pubmed
fi
touch .fetched_data
