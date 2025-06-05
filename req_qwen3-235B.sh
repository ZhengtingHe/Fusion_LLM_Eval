curl --location  http://180.213.184.177:30084/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
    "model":"qwen3-235B",
    "messages": [{'role': 'user', 'content': '你是谁？'}],
    "max_tokens": 5000,
    "stream": true
}'