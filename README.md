# Instamart Customer Intelligence & Support Copilot

## Overview
A comprehensive Copilot built for Swiggy Instamart to clean synthetic order data, segment customers, predict lapse probabilities, automatically triage support tickets, and resolve refunds using a RAG-based AI Agent.

## Reproduce
1. `python -m venv venv`
2. `source venv/bin/activate` (or `venv\Scripts\activate` on Windows)
3. `pip install -r requirements.txt`
4. `python generate_data.py`
5. `python part_a_cleaning.py`
6. `python part_b_models.py`
7. `python part_c_triage.py`
8. `python part_d_agent.py`

## Part A
* **Raw Audit:** 13,084 order rows; 119 exact duplicates; 656 pending rows; 12,309 distinct order_id; 120 unparseable amounts; 106 negative delivery times; 2,541 blank delivery times. Customers: 61 blank ages; 6 outside 10-100; 24 distinct raw city spellings; 10 raw gender spellings; 30 blank genders.
* **Cleaning Rules:**
  1. Converted `order_ts` to datetime.
  2. Removed 119 exact duplicate rows.
  3. Kept latest `order_ts` per `order_id` (removed 656 earlier pending records).
  4. Filled 120 missing/unparseable `order_amount` values using price calculations.
  5. Imputed 2,541 missing `delivery_minutes` for completed orders using category median.
* **SQL Results:** 
  - Q1: 30 customers.
  - Q2: 30 rows, Bengaluru/Electronics highest at 233,909.45.
  - Q3: 11 products.
  - Q4: 25 customers.
  - Q5: 550 customers above the 2,552.06 average.
  - Q6 Ties: RANK() skips ranks after ties, whereas ROW_NUMBER() sequentially assigns unique values, causing them to differ when customers have identical spend.
  - Q7: 183 daily rows ending in 2,998,668.80.
* **pandas Cross-Check:** The dataset records grocery transactions with prices skewed heavily rightward (mean 269.91 vs median 135.00), showing 897 upper outliers above the 645.30 upper fence. The largest absolute correlation is between `order_amount` and `discount_pct` (-0.095). A correlation of this size indicates a weak inverse relationship, but correlation does not establish causation.

## Part B
* **RFM Summary:** 1,175 customers. Recency median 19; Frequency mean 9.46; Monetary mean 2,552.06.
* **K-Means Clustering:**
  - K=2: 1820.8 / 0.465
  - K=3: 1268.2 / 0.445
  - K=4: 743.1 / 0.526
  - K=5: 617.4 / 0.481
  - K=6: 503.2 / 0.482
  - K=7: 408.5 / 0.416
  - K=8: 366.5 / 0.412
  - *Note:* Inertia continually drops as K increases because adding more clusters always reduces distance to centroids, yet this risks overfitting; hence, inertia alone cannot dictate K.
* **Preprocessing:** No scaler or one-hot encoder saw the test data. The full cohort mean differs from the scaler's learned mean because the scaler was strictly fit only on the training set to prevent data leakage.
* **Model Choice:** The tuned Random Forest model was selected (Max Depth 8, Min Samples Leaf 10) achieving an F1 score of ~0.65 on the lapsed class. Over-sampling increased recall from ~0.64 to ~0.85 while precision stayed near ~0.60. Missing a lapsing customer means lost revenue, while wrongly targeting a loyal one only costs the price of a small discount campaign; therefore, optimizing for higher recall is much more cost-effective for the retention team.

## Part C
* **Guardrails:** 278 PII elements replaced (55 emails, 91 phone numbers, 132 order ids). 
* **Model Triage:** 135 `classified_json`, 12 `classified_keyword`, 9 `flagged_invalid_output`.
* **Zero-Shot vs Few-Shot:** Zero-shot accuracy was 85.7%, while few-shot achieved 100%. The few-shot prompt uses roughly 25-30 extra tokens per ticket. The extra tokens are worth paying for because the few-shot prompt completely eliminates critical routing errors (like sending Fraud tickets to the Billing queue).
* **Validation:** 51 tickets triggered `budget_exhausted` in the demo. In the final run, explicitly rule-validated tickets were assigned routes; invalid LLM outputs were routed to `human_review`.

## Part D
* **Retrieval Evaluation:** 12/12 expected_doc retrieved, 12/12 expected_fact found, 2/2 unanswerable handled. With 0 overlap, questions fail because the expected fact is split across a boundary.
* **TF-IDF limitations:** TF measures word frequency in a document, and IDF measures word rarity across the corpus. A limitation is that it fails at semantic matching; synonyms won't match if they don't share exact substrings.
* **Agent Results:**
| Scenario | Status | Refund Amount | LLM Calls |
|---|---|---|---|
| S1 | REFUND | 128.25 | 5 |
| S2 | REFUND | 135.00 | 5 |
| S3 | ESCALATE | 1999.00 | 5 |
| S4 | DENIED | 0.00 | 4 |
| S5 | DENIED | 0.00 | 4 |
| S6 | NOT_FOUND | 0.00 | 2 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 |
* **Stop Safely:** Token Limit Demo Status: `FALLBACK_TOKEN_LIMIT`. Timeout Demo Status: `FALLBACK_TIMEOUT`.
* **Resume Demo Tools:** `get_order`, `search_policy`, `calculate_refund`, `issue_refund`.
* **ReAct vs Plan-and-Execute:** ReAct required up to 5 model calls for a successful refund (S1), while simpler failures like NOT_FOUND (S6) resolved in 2 calls. A plan-and-execute model would ask for the entire 4-step execution plan upfront, potentially saving tokens on straightforward tickets. However, ReAct is vastly superior for this refund workflow because it observes the environment at each step. If `get_order` returns an unexpected "cancelled" status or "not found", ReAct immediately halts and denies the request, whereas a rigid plan-and-execute system might blindly proceed to calculate a refund on bad data unless complex, error-handling conditional logic is explicitly engineered into the planner.

## Limitations and Next Steps
The Premium flag is a post-hoc whole-window label derived from September data, meaning it leaks future knowledge into the cohort. Consequently, the retention list is illustrative and not a strict leakage-free evaluation. Next steps include migrating to dense embeddings for better semantic retrieval and implementing a hosted LLM to handle complex reasoning at scale.