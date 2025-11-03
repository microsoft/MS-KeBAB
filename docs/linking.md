# Linking task

The linking task is to decide if two entity fragments come from the same entity or not.  An entity fragment is represented using the Entity class and serialized as json.  The label is True if the fragments come from the same entity, and False otherwise.  Predictions should be provided as the log-odds of the label being True, i.e. the log-probability of True minus the log-probability of False. Predictions can also be boolean but this prevents computing the log-probability metric.  

# Metrics

The main metric for ordering different linking systems is the average log-probability of the correct label.  This the main metric because it rewards systems for being appropriately uncertain (it is a so-called "proper scoring rule").  Secondary metrics include precision (how often a True prediction was correct) and recall (how many True pairs were labelled True).  These metrics ignore the uncertainty in the predictions.

# Debugging

A detailed breakdown of results on the task is output to the file 'linking_predictions.tsv'.  Each row is a single linking pair.  The columns have various properties of the pair that can be used for filtering the rows.

| Column name | Description |
|---|---|
| *left* and *right* | the two entity fragments to link, in json format |
| *entity_type* | A list of type strings, the union of left and right |
| *overlap_props* | A list of the overlapping property keys |
| *prop_overlap_num* | The length of overlap_props |
| *prop_pattern* | A pair of tuples, containing the property keys of left and right |
| *name_overlap* | TRUE iff left and right have a common name |
| *label* | The ground truth label |
| *log_odds* | The prediction |
| *predicted_label* | TRUE iff log_odds > 0 |
| *left_entity_id* and *right_entity_id* | The WikiData id of the entities |
| *debugging_info* | Linker-specific information provided to the evaluate method via debugging_info_path |