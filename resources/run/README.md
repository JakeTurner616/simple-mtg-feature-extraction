## Run directory

- Holds feature descriptors
- Card label, name, and id mappings within sqlite db
- faiass_ivf and index_to_card.txt for feature searching

# Files location / How to aquire

- Pre trained files for building from source can be aquired for free via [kaggle](https://kaggle.com/datasets/15870d18aa824bb278497b1ddc51e1d6183e01211174eff852b1aa63089048d6) or [hf](https://huggingface.co/datasets/JakeTurner616/mtg-cards-SIFT-Features/tree/main) Storage requirements are 2.3 GB

- files can also be aquired by running the workflow. Storage requirements are >122 GB for all card scans + final files. 

- pre-trained files will be downloaded at runtime from huggingface on first launch in the pre-compiled windows binary.
