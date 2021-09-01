# Obtaining similar dialogs using embeddings

Given two dialogs, this code is capable of computing their dialog embeddings and compare them in terms of cosine similarity. We used the Sentence-BERT (SBERT) approach proposedby Reimers and Gurevych (2019), which is a modification of the pre-trained BERT network that uses siamese  and triplet network structures to derive semantically meaningful sentence embeddings that can be compared using cosine-similarity. 

### Bibliography

Reimers, N.; and Gurevych, I. 2019.  Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.  In *Proceedings of the 2019 Conference on  Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*,  3982–3992.  Hong  Kong,  China:  Association  forComputational Linguistics.