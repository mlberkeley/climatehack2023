python query_scoreboard.py

curl -X POST -H 'Content-type: application/json' --data "{
	\"blocks\": [
        {
			\"type\": \"section\",
			\"text\": {
				\"type\": \"mrkdwn\",
				\"text\": \" :loud_sound: *LEADERBOARD* :trophy:\"
			}
		},
        {
			\"type\": \"divider\"
		},
        {
			\"type\": \"section\",
			\"text\": {
				\"type\": \"mrkdwn\",
				\"text\": \"$(cat scoreboard.txt)\"
			}
		}
    ]
}" https://hooks.slack.com/services/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
