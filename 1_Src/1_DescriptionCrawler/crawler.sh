#!/bin/bash

# Read the package names, remove carriage returns, and create a temporary JSON array
tr -d '\r' < pkgNames.txt | jq -Rn '[inputs]' > pkgNames.json

# Process the JSONL file and match package names with descriptions
jq --argfile pkgNames pkgNames.json '
  . as $item |
  $pkgNames[] |
  select($item.docid == .) |
  "\($item.docid),\($item.descriptionHtml)"
' gp-metadata-full.jsonl > output.txt

# Clean up temporary JSON array
rm pkgNames.json

echo "Processing complete. Output saved to output.txt"
