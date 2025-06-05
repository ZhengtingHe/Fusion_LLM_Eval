curl --location 'http://180.213.184.177:30084/v1/chat/completions' \
--header 'Content-Type: application/json' \
--data '{
    "model": "DeepSeek-R1-0528-AWQ",
    "stream": true,
    "messages": [{"role": "user", "content": "你是谁？"}]
}'