Using these datasets was inspired by the following paper:
M. Zehlike and C. Castillo, “Reducing disparate exposure in ranking: A learning to rank approach.” [Online]. Available: http://arxiv.org/pdf/1805.08716v1

Used features and describtion:

|  Name | Description |
| ----- | ----------- |
|  query_id | The query id of an item. |
| gender | The protected attribute; 0 if male; 1 if female |
| match_body_email_subject_score_norm | The normalized score from top matches of query terms found inside the mail's body, email, and subject. |
| match_body_email_subject_df_stdev | The raw standard deviation of the document frequency in the document results which match the query terms in the mail's body, email, and subject. |
| match_body_email_subject_idf_stdev | The classic standard deviation of the inverse document frequency in the results of the documents which match the query terms in the mail's body, email, and subject. |
| match_body_score_norm | The normalized search score of the results from a search through the mail's body. |
| match_subject_score_norm | The normalized search score of the results from a search through the mail's subject. |
| judgement | A number indicating the rank of the item with regard to the query id. |