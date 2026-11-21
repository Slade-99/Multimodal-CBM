New Approach:

1) Dataset Fixed
2) Codes in the Debugging file is used to first train the encoders for doing good at concept prediction task.
3) Their results are stored at cxr_concept_checkpoints , ecg_concept_checkpoints ..... etc
4) Next step is to use the weights of those pretrained encoders and train for the downstream task, codes are in folder Stage_2
5) 