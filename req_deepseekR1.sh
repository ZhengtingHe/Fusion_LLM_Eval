curl --location http://180.213.184.147:30082/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
    "model": "deepseek-r1",
    "messages": [
        {
            "role": "user",
            "content": "给我讲个笑话"
        }
    ],
    "max_tokens": 12000,
    "temperature": 0.7,
    "stream": false
}'