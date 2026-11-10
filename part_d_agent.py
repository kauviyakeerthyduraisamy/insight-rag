import os
import json
import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mock_llm
import sqlite3
from datetime import datetime

os.makedirs('state', exist_ok=True)

# 1. Chunk Documents
chunks = []
for file in os.listdir('data/policies'):
    if not file.endswith('.txt'): continue
    with open(f'data/policies/{file}', 'r') as f:
        words = f.read().split()
        for i in range(0, len(words), 30):
            chunk_words = words[i:i+40]
            chunks.append({'source': file, 'text': ' '.join(chunk_words)})
            if i + 40 >= len(words): break

# 2. Retrieval setup
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([c['text'] for c in chunks])

def retrieve(query, k=3):
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, tfidf_matrix).flatten()
    top_k = sims.argsort()[-k:][::-1]
    return [chunks[i] for i in top_k]

# 3. Tools Definitions
def get_order(order_id):
    conn = sqlite3.connect('data/instamart.db')
    df = pd.read_sql_query(f"SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id='{order_id}'", conn)
    conn.close()
    if df.empty: return {"error": "order not found"}
    return df.iloc[0].to_dict()

def search_policy(query):
    return {"results": retrieve(query)}

def calculate_refund(order_amount, delivery_minutes):
    if pd.isna(delivery_minutes) or int(delivery_minutes) <= 45: return {"refund_inr": 0}
    return {"refund_inr": float(order_amount)}

def log_ledger(tool, order_id, refund_inr):
    with open('outputs/refund_ledger.jsonl', 'a') as f:
        f.write(json.dumps({"ts": datetime.now().isoformat(), "action": tool, "order_id": order_id, "amount": refund_inr}) + '\n')
    return {"refund_inr": refund_inr}

TOOL_MAP = {
    'get_order': get_order,
    'search_policy': search_policy,
    'calculate_refund': calculate_refund,
    'issue_refund': lambda order_id, refund_inr: log_ledger('issue_refund', order_id, refund_inr),
    'escalate_to_human': lambda order_id, refund_inr: log_ledger('escalate_to_human', order_id, refund_inr)
}

# 4. Agent Loop
def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10, run_id=None):
    mock_llm.reset()
    start_time = time.monotonic()
    messages = [{"role": "system", "content": "AGENT: Refund assistant"}, {"role": "user", "content": user_message}]
    tokens_used = 0
    executed_tools = []
    
    for step in range(max_iterations):
        if time.monotonic() - start_time >= time_limit_seconds:
            return {"status": "FALLBACK_TIMEOUT", "final_answer": "Timeout reached, human handoff.", "tokens_used": tokens_used}
            
        resp = mock_llm.chat(messages, tools=True)
        tokens_used += resp.usage.total_tokens
        
        if tokens_used >= token_limit:
            return {"status": "FALLBACK_TOKEN_LIMIT", "final_answer": "Token limit reached, human handoff.", "tokens_used": tokens_used}
            
        msg = resp.choices[0].message
        if not getattr(msg, 'tool_calls', None):
            ans = msg.content
            return {"status": ans.split(':')[0] if ':' in ans else "UNKNOWN", "final_answer": ans, "tokens_used": tokens_used, "steps": executed_tools}
            
        tc = msg.tool_calls[0]
        messages.append({"role": "assistant", "content": None, "tool_calls": [tc.__dict__]})
        
        tool_res = TOOL_MAP[tc.name](**tc.arguments)
        executed_tools.append({'name': tc.name, 'arguments': tc.arguments, 'result': tool_res})
        
        messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": json.dumps(tool_res)})
        
        if run_id:
            with open(f'state/{run_id}.json', 'w') as f:
                json.dump({"run_id": run_id, "messages": messages, "tokens_used": tokens_used, "tools": executed_tools}, f)

    return {"status": "FALLBACK_MAX_ITERATIONS", "final_answer": "Max steps reached, human handoff.", "tokens_used": tokens_used}

if __name__ == "__main__":
    scenarios = pd.read_csv('data/raw/agent_scenarios.csv')
    with open('outputs/agent_runs.jsonl', 'w') as f:
        for _, row in scenarios.iterrows():
            res = run_agent(row['message'], row['scenario_id'])
            res['scenario_id'] = row['scenario_id']
            f.write(json.dumps(res) + '\n')
            
    # Timeout/Token Demos
    run_agent("Refund ORD-123456", "S_TOKEN_DEMO", token_limit=200)
    run_agent("Refund ORD-123456", "S_TIME_DEMO", time_limit_seconds=0)