Add more dropouts to help overfitting

Use more pooling layers - can't until the reached the benchmark

Confirm testing is being done right. validation with fnames - DONE

Implement the combining of data streams for TSCNN
 - use pickle to store the logits.
 - load the logits and softmax them
 - find the average
 - calculate TSCNN

Maybe, implement them simultaneously and get graphs for the accuracy, to plot LMC, MC and TSCNN on the same accuracy and loss graph

----------------------------------
Notes:
- Removed the bias terms from conv1-4 since batchnormalisation is used (something about it has its own bias terms). This got the TensorSummary Parameters identical to that of Table 1 in the report
- Used Adam with L2 regularisation using weight decay
