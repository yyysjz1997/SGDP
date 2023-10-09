## SGDP: A Stream-Graph Neural Network Based Data Prefetcher [[Paper]](https://arxiv.org/abs/2304.03864)

### Introduction of Code Demo and Appendix

There are seven python files in this folders. 

- cache.py: A cache simulator file, which includes a class about cache for cache initialization and execution.
- cachetest.py:  Cache test for single-step and rolling prefetching, which includes data pre-processing and top-K encoding.
- data_process.py: Complete data processing file for both MSRC, HW and other datasets. It will save the trace as the stream form for model training and model test.
- main.py: Main function of our SGDP data prefetching framework. We can train and test the SGDP and its variants by **python main.py** based on default parameters.
- model.py: Our stream-graph neural network model and  the subfunction for training, testing and prediction.
- pred_list_gene.py: Generate the local and global connection matrix and hybrid connection matrix. And executive the single-step and rolling data prefetching, output the HR and EPR for corresponding input trace.
- utils.py: Other functions for data processing and model building. Besides, there is a function for global stream-graph structure.

### Contributions

1. SGDP can accurately learn complex access patterns by capturing the relations of LBA deltas in each stream. The relations are represented by sequential connect matrices and full-connect matrices using graph structures.

2. To the best of our knowledge, SGDP is the first work that utilizes the stream-graph structure of the LBA delta in the data prefetching problem. Using gated graph neural networks and attention mechanisms, we extract and aggregate sequential and global information for better prefetching.

3. As a novel solution in the hybrid storage system, SGDP can be generalized to multiple variants by different stream construction methods, which further enhances its robustness and expands its application to various real-world scenarios. 

4. SGDP outperforms SOTA prefetchers by 6.21\% on hit ratio, 7.00\% on effective prefetching ratio, and speeds up inference time by 3.13X on average. It has been verified in commercial hybrid storage systems in the experimental phase and will be deployed in the future product series.
