/*
First repacled all values of " " (blank space) with "" (empty string):

sed -i -e 's/ //g' data/HMXPC12_DI_v2_5-14-14.csv

*/

-- copy csv files into table
COPY harvard_data
FROM '/Users/jeremymiller/GoogleDrive/Data_Science/Projects/Education_data/harvard_ed_x/data/HMXPC13_DI_v2_5-14-14.csv' WITH (FORMAT CSV, HEADER, FORCE_NULL(grade));
