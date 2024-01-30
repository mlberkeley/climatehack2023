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
				\"text\": \"$(cat /home/gracetang/climatehack2023/misc/scoreboard.txt)\"
			}
		}
    ]
}" https://hooks.slack.com/services/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