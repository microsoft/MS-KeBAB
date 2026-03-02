# Clustering task

The clustering task is to divide a set of entity fragments into clusters.  An entity fragment is represented using the Entity class and serialized as json.  A clustering is represented by tagging each fragment with a cluster id, which can be any string excluding newlines.

# Metrics

A predicted clustering is compared to the ground truth clustering using BCubed precision, recall, and F1 metrics.  BCubed precision is the average over fragments of the fraction of other fragments in its predicted cluster that also appear in its ground-truth cluster.  BCubed recall is the average over fragments of the fraction of other fragements in its ground-truth cluster that also appear in its predicted cluster.  F1 combines the two.
