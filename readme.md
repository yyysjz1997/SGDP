### Introduction of Code Demo and Appendix

There are seven python files in this folders. 

- cache.py: A cache simulator file, which includes a class about cache for cache initialization and execution.
- cachetest.py:  Cache test for single-step and rolling prefetching, which includes data pre-processing and top-K encoding.
- data_process.py: Complete data processing file for both MSRC, HW and other datasets. It will save the trace as the stream form for model training and model test.
- main.py: Main function of our SGDP data prefetching framework. We can train and test the SGDP and its variants by **python main.py** based on default parameters.
- model.py: Our stream-graph neural network model and  the subfunction for training, testing and prediction.
- pred_list_gene.py: Generate the local and global connection matrix and hybrid connection matrix. And executive the single-step and rolling data prefetching, output the HR and EPR for corresponding input trace.
- utils.py: Other functions for data processing and model building. Besides, there is a function for global stream-graph structure.
- Appendix.pdf: The Supplementary Material for SGDP. It includes many details results and introduction of algorithm reproducing.

