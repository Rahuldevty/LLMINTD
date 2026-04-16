import requests
services = [
    ('planner','http://localhost:8001/planner/analyze', {'prompt':'Hello'}),
    ('researcher','http://localhost:8002/researcher/rewrite', {'prompt':'Hello'}),
    ('generator','http://localhost:8003/generator/respond', {'prompt':'Hello'}),
    ('verifier','http://localhost:8004/verifier/check', {'prompt':'Hello','response':'Hello'}),
    ('api','http://localhost:8000/health', None)
]
for name,url,payload in services:
    try:
        if name=='api':
            r=requests.get(url,timeout=20)
        else:
            r=requests.post(url,json=payload,timeout=20)
        print(name, r.status_code, r.text[:200])
    except Exception as e:
        print(name, 'ERROR', type(e).__name__, e)
