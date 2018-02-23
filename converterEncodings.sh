#!/bin/bash

# Recursive file convertion iso-8859-1 --> utf-8
# Place this file in the root of your site, add execute permission and run
# Converts *.php, *.html, *.css, *.js files.
# To add file type by extension, e.g. *.cgi, add '-o -name "*.cgi"' to the find command

#find ./ -name "*.php" -o -name "*.html" -o -name "*.css" -o -name "*.js"  -type f |
find ./ -name "*.html" -type f |
while read file
do
  echo " $file"
  mv $file $file.icv
  iconv -f ISO-8859-1 -t UTF-8 $file.icv > $file
  rm -f $file.icv
done
