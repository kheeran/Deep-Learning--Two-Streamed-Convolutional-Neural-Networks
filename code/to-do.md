Add more dropouts to help overfitting

Use more pooling layers

Implement the combining of data streams



----------------------------------
Notes:
- Removed the bias terms from conv1-4 since batchnormalisation is used (something about it has its own bias terms). This got the TensorSummary Parameters identical to that of Table 1 in the report
- Used Adam with L2 regularisation using weight decay

