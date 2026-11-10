import pandas as pd
import mock_llm
import time
import random
import re
import json

# Prompts
ZERO_SHOT_PROMPT = "CLASSIFY the following support ticket into one of these categories: billing, delivery, technical, fraud. Return ONLY valid JSON in the format: {'category': '...', 'reason': '...'}"
FEW_SHOT_PROMPT = ZERO_SHOT_PROMPT + "\nEXAMPLES:\nTicket: 'My app crashed' -> {'category': 'technical', 'reason': 'app issue'}\nTicket: 'Unauthorized card transaction' -> {'category': 'fraud', 'reason': 'unauthorized charge'}"

def redact_text(text):
    text = re.sub(r'[\w.+-]+@[\w-]+\.[\w.]+', '[EMAIL]', text)
    text = re.sub(r'\b[6-9]\d{9}\b', '[PHONE]', text)
    text = re.sub(r'ORD-\d{6}', '[ORDER_ID]', text)
    return text

def call_mock_llm_with_retry(messages):
    BASE_DELAY = 0.05
    for attempt in range(4):
        try:
            return mock_llm.chat(messages)
        except mock_llm.TransientLLMError:
            if attempt == 3: raise
            wait = BASE_DELAY * (2 ** attempt) * random.uniform(0.9, 1.15)
            print(f"Retry {attempt} Waiting {wait:.4f}s")
            time.sleep(wait)

def run_triage(tickets, prompt, budget_limit=None):
    mock_llm.reset()
    results = []
    tokens_used = 0
    
    for _, row in tickets.iterrows():
        text = str(row['text'])
        status, cat = "", ""
        
        if not text.strip():
            status = 'skipped_empty'
        elif mock_llm.est_tokens(text) > 400:
            status = 'flagged_long'
        else:
            redacted = redact_text(text)
            messages = [{"role": "system", "content": prompt}, {"role": "user", "content": redacted}]
            
            prompt_est = mock_llm.est_tokens(prompt) + mock_llm.est_tokens(redacted)
            if budget_limit and (tokens_used + prompt_est >= budget_limit):
                status = 'budget_exhausted'
            else:
                resp = call_mock_llm_with_retry(messages)
                tokens_used += resp.usage.total_tokens
                reply = resp.choices[0].message.content
                
                try:
                    parsed = json.loads(reply)
                    if parsed.get('category') in ['billing', 'delivery', 'technical', 'fraud']:
                        status = 'classified_json'
                        cat = parsed['category']
                    else:
                        status = 'flagged_invalid_output'
                except:
                    words = re.findall(r"[a-z]+", reply.lower())
                    valid_cats = [c for c in ['billing', 'delivery', 'technical', 'fraud'] if c in words]
                    if len(valid_cats) == 1:
                        status = 'classified_keyword'
                        cat = valid_cats[0]
                    else:
                        status = 'flagged_invalid_output'
                        
        results.append({
            'ticket_id': row['ticket_id'], 'customer_id': row['customer_id'],
            'status': status, 'category': cat, 'tokens_used': tokens_used
        })
    return pd.DataFrame(results)

if __name__ == "__main__":
    tickets = pd.read_csv('data/raw/tickets.csv', keep_default_na=False)
    segments = pd.read_csv('outputs/customer_segments.csv')
    
    # Budget Demo
    demo_res = run_triage(tickets, ZERO_SHOT_PROMPT, budget_limit=5000)
    demo_res.to_csv('outputs/triage_budget_demo.csv', index=False)
    print(f"Budget Exhausted Tickets: {(demo_res['status'] == 'budget_exhausted').sum()}")
    
    # Final Routing
    final_res = run_triage(tickets, FEW_SHOT_PROMPT, budget_limit=100000)
    final_res = final_res.merge(segments[['customer_id', 'segment_name', 'is_premium']], on='customer_id', how='left')
    final_res['is_premium'] = final_res['is_premium'].fillna(False)
    
    def apply_routing(row):
        if row['status'] in ['skipped_empty', 'flagged_long', 'flagged_invalid_output', 'budget_exhausted']:
            return pd.Series(['human_review', 'NORMAL'])
        if row['category'] == 'fraud': return pd.Series(['security_team', 'HIGH'])
        if row['is_premium']: return pd.Series(['premium_desk', 'HIGH'])
        return pd.Series(['support_queue', 'NORMAL'])
        
    final_res[['route', 'priority']] = final_res.apply(apply_routing, axis=1)
    final_res.to_csv('outputs/triage_results.csv', index=False)