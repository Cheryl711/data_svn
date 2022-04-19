# IVN


## Semi-automatic Service Value Network Modeling Approach based on External Public Data


* Interface（Flask）  
    > server1.py 
* ***Domain entity recognition algorithm***
    > /ast_html/classtification.py     *[html_classify()]*
    * Data
        > /ast_html/html_labeled_0428updata.json
* ***Domain relationship extraction algorithm***
    > /extraction/mapping.py      *[classify_whole()/classify_single()]*
    
    > /classification/classify.py      *[get_scope_classify()]*
    * Data
        > 36Kr_history_encoded.txt
* ***DVC extraction algorithm***
    > /extract/embedding.py
    
    > /extract/extract_chain.py
    * Data
        > /extract/data/*
* Generate IVN
    > /classification/generate_abs.py

