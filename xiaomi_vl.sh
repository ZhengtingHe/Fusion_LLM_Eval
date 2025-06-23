curl -X POST "http://180.213.184.182:30083/v1/chat/completions" \
	-H "Content-Type: application/json" \
	--data '{
		"model": "XiaomiMiMo/MiMo-VL-7B-RL",
		"messages": [
			{
				"role": "user",
				"content": [
					{
						"type": "text",
						"text": "用一句话描述这张图"
					},
					{
						"type": "image_url",
						"image_url": {
							"url": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
						}
					}
				]
			}
		]
	}'