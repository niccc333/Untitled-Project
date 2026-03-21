import requests
import re

GAMMA_BASE = "https://gamma-api.polymarket.com"

params = {
    "limit": 100,
    "active": "true",
    "closed": "false",
}

print("Paginating markets without query param...")
solana_markets = []

for offset in range(0, 5000, 100):
    params["offset"] = offset
    resp = requests.get(f"{GAMMA_BASE}/markets", params=params)
    data = resp.json()
    if not data:
        break
        
    for m in data:
        q = m.get("question", "")
        if re.search(r'\b(solana|sol)\b', q, re.IGNORECASE):
            if "solid" not in q.lower() and "solve" not in q.lower() and "solar" not in q.lower():
                solana_markets.append(m)

print(f"Total Solana markets found locally: {len(solana_markets)}")
for m in solana_markets[:10]:
    print(f" - {m.get('question')}")
