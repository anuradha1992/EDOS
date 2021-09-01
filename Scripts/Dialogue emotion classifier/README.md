# EmoBERT+

### Fine-grained dialog emotion classifier for movie dialogs.

Given a dialog utterance, the classifier annotates it with one out of a list of labels containing 32 emotions, 8 response intents, and neutral. The classifier is trained and tested on human annotated movie dialogs taken from OpenSubtitles 2018 corpus (Lison et al., 2019).  

The pretrained weights necessary to initiate the model prior to training can be downloaded from: [drive.google.com/drive/folders/1KvTt1aK2a2JFKR_YcGaQnc_eNKW-qAos?usp=sharing](https://drive.google.com/drive/folders/1KvTt1aK2a2JFKR_YcGaQnc_eNKW-qAos?usp=sharing)

### Datasets

All the datasets necessary to train and test the classifier are included in the folder ../../Data/Training data for EmoBERT+ included in the Data Appendix

It contains the following datasets:

1. ***MTurk_groundtruth_labels_9K.csv***: Human annotated ground truth labels obtained from MTurk. These are dialogs for which 2 out of 3 workers agreed on the same label.
2. ***Similar_dialogs_3K.csv***: These are dialogs that are semantically similar to dialogs with ground truth labels. The similar dialogs are obtained by comparing their embeddings using cosine similarity.  
3. ***Self-labeled_dialogs_4K***: These are dialogs automatically labeled by the EmoBERT+ classifier trained in the 1st iteration using the above training datasets. This contains the top-100 high confidence utterances in each category along with their proceeding context.
4. ***Similar_self-labeled_dialogs_2K***: These are dialogs that are similar to the self-labeled dialogs obtained by comparing their embeddings using cosine similarity.

### Dependencies

Following are the versions of python packages the code has been tested on.

- numpy (1.18.2)
- tensorflow (2.2.0rc2)
- tensorflow-gpu (2.1.0)
- pytorch-transformers (1.2.0)
- h5py (2.10.0)
- pyyaml (3.13)

### Bibliography

Lison, P.; Tiedemann, J.; Kouylekov, M.; et al. 2019.  Opensubtitles 2018: Statistical rescoring of sentence alignments inlarge, noisy parallel corpora.  In *LREC 2018, Eleventh Inter-national Conference on Language Resources and Evaluation*. European Language Resources Association (ELRA).

Hannah Rashkin, Eric Michael Smith, Margaret Li and Y-Lan Boureau. 2019.  Towards Empathetic Open-domain Conversation  Models:  A  New  Benchmark  and  Dataset.   In *Proceedings  of  the  57th  Annual  Meeting  of  the Association for Computational Linguistics*, pages 5370–5381, Florence, Italy.

Devlin, J.; Chang, M.-W.; Lee, K.; and Toutanova, K. 2019. BERT: Pre-training of Deep Bidirectional Transformers forLanguage Understanding.  In *Proceedings of the 2019 Conference of the North American Chapter of the Association forComputational Linguistics: Human Language Technologies*, Volume 1 (Long and Short Papers), 4171–4186. Minneapolis, Minnesota: Association for Computational Linguistics.
