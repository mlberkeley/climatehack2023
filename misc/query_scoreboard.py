import requests
from bs4 import BeautifulSoup

url = 'https://doxaai.com/competition/climatehackai-2023/scoreboard'
html_text = requests.get(url).text
soup = BeautifulSoup(html_text, 'html.parser')
filepath = "scoreboard.txt"
with open(filepath, 'w') as file:
    file.write("")

rank = 1
for contestant in soup.find_all(attrs={"class": "Scoreboard_scoreboardRow__Qdshf"}, limit=4)[1:4]:
    # for c in contestant.find_all(attrs={"class": "Link_userLinkUsername__ELc34"}):
    #     print(c.text, end="; ")
    result = "| " + str(rank) + ". "
    
    i = 0
    for field in contestant:
        if i == 0 or i == 3:
            i += 1
            continue
        if i == 1:
            result += "user: "
        elif i == 2:
            result += "school: "
        elif i == 4:
            result += "mae: "
        result += "`"
        result += field.text
        result += "` | "
        i += 1  
    result += "\n-------------------------------------------------------------\n"    
    with open(filepath, 'a') as file:
            file.write(result)
    rank += 1

first_line = ""
with open(filepath, 'r') as file:
    first_line = file.readline()

with open(filepath, 'a') as file:
    if "UC Berkeley" in first_line:
        file.write("we're in first. *gob ears!*")
    else:
        file.write("we're not in first. what gives?")