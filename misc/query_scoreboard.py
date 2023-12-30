import requests
from bs4 import BeautifulSoup
import subprocess
import json

winners = {}
danger_thresh = 0.1
url = 'https://doxaai.com/competition/climatehackai-2023/scoreboard'
score_filepath = "/home/gracetang/climatehack2023/misc/scoreboard.txt"
winners_filepath = "/home/gracetang/climatehack2023/misc/winners.json"
commandpath = "/home/gracetang/climatehack2023/misc/send_slack.sh"
scoreboard_row_str = "Scoreboard_scoreboardRow__Qdshf"
best_school = "UC Berkeley"

def danger():
    for winner in winners:
        if winners[winner][0] != best_school and float(winners[winner][1]) < danger_thresh:
            return True
    return False

def scoreboard_changed(winners_filepath, score_filepath):
    old_winners = {}
    
    with open(winners_filepath, 'r') as winners_file:
        old_winners = json.load(winners_file)
        
    if winners == old_winners:
        return False
    
    # we don't really care about movement here unless we're no longer first 
    # or unless someone passes the threshold       
    if set(winners.keys()) == set(old_winners.keys()) and not danger():
        if winners[next(iter(winners))][0] == best_school:
            return False
        
    with open(score_filepath, 'w') as file:
        if danger():
            file.write(":rotating_light: *A CHALLENGER APPROACHES* :rotating_light:\n")
        else:
            file.write("")
        
    return True

def generate_scoreboard(scoreboard_row_str):
    result = ""
    for contestant in soup.find_all(attrs={"class": scoreboard_row_str}, limit=6)[1:6]:
        i = 0
        name = ""
        for field in contestant:
            if i == 0:
                result += "| "
                result += field.text
                result += ". "
                i += 1
                continue
            elif i == 1:
                result += "user: "
                name = field.text
                winners[name] = []
            elif i == 2:
                result += "school: "  
                winners[name].append(field.text)
            elif i == 3:
                i += 1
                continue
            elif i == 4:
                result += "mae: "
                winners[name].append(field.text)
            result += "`"
            result += field.text
            result += "` | "
            i += 1  
        result += "\n----------------------------------------------------------------------\n" 
    return result
    
def add_comment(score_filepath):
    with open(score_filepath, 'a') as file:
        school = winners[next(iter(winners))][0]
        if school == best_school:
            file.write("we're in first. *gob ears!* :bear:\n")
        else:
            file.write("we're not in first. what gives? :3nking_face:\n")
            
def run_slack_send(commandpath):
    with open(commandpath, 'r') as file:
        bash_command = file.read()
        subprocess.run(bash_command, shell=True, capture_output=True, text=True)
    
if __name__ =="__main__":
    html_text = requests.get(url).text
    soup = BeautifulSoup(html_text, 'html.parser')
    
    result = generate_scoreboard(scoreboard_row_str)
    
    if scoreboard_changed(winners_filepath, score_filepath):
        with open(score_filepath, 'a') as file:
            file.write(result)
        with open(winners_filepath, 'w') as file:
            json.dump(winners, file)   
        add_comment(score_filepath)
        run_slack_send(commandpath)