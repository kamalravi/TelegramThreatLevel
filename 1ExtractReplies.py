import json
import datetime
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter
import matplotlib.ticker as ticker

import glob
from natsort import natsorted
import pandas as pd


filePath = f'/home/ravi/raviProject/DATA/DataCollection/'


chatfolders=natsorted(glob.glob(filePath+'/*'))


for idx, folder in enumerate(chatfolders):

    print(idx+1,folder)
    
    # data = pd.read_json(folder+'/result.json', orient='records')

    json_file_path = folder # + '/result.json'

    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)
    except json.decoder.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        with open(json_file_path, 'r') as file:
            lines = file.readlines()
            error_line_number = e.lineno
            error_line = lines[error_line_number - 1].strip()
            print(f"Problematic line {error_line_number}: {error_line}")
            print("Context around the error:")
            for i in range(max(0, error_line_number - 2), min(len(lines), error_line_number + 1)):
                print(f"{i + 1}: {lines[i].strip()}")

    channelName = str.split(folder.split('/')[-1],'.')[0]

    messages = {}

    for message in data['messages']:
        if message["type"] == "message":
            if "reply_to_message_id" in message:
                
                parent_id = message["reply_to_message_id"]
                if parent_id in messages:
                    messages[parent_id]["replies"].append(
                        {
                        "id": message["id"],
                        "text": message["text"],
                        "date": message["date"],
                        "date_unixtime": message["date_unixtime"]
                        }
                    )
                    messages[parent_id]["reply_count"] += 1

            else:
                messages[message["id"]] = {
                    "text": message["text"],
                    "date": message["date"],
                    "date_unixtime": message["date_unixtime"],
                    "replies": [],
                    "reply_count": 0
                }


    msg_counts = {}
    reply_counts = {}

    for message_id, message_data in messages.items():
        msg_date = datetime.datetime.utcfromtimestamp(int(message_data["date_unixtime"])).date()
        if msg_date not in msg_counts:
            msg_counts[msg_date] = 0
        
        if msg_date not in reply_counts:
            reply_counts[msg_date] = 0

        msg_counts[msg_date] += 1

        for reply in message_data["replies"]:
            reply_date = datetime.datetime.utcfromtimestamp(int(reply["date_unixtime"])).date()
            if reply_date not in reply_counts:
                reply_counts[reply_date] = 0
            reply_counts[reply_date] += 1



    dates = sorted(msg_counts.keys())
    total_values = []
    
    for date in dates:
        total = msg_counts[date] + reply_counts[date]
        msg = msg_counts[date]
        ratio = total / msg
        #print(ratio)
        total_values.append(ratio)
    #msg_values = [msg_counts[date] for date in dates]
    #reply_values = [reply_counts[date] for date in dates]
    print(max(total_values))
    print(min(total_values))
    fig, ax = plt.subplots(figsize = (10,10))
    #ax.plot(dates, msg_values, label="Messages")
    #ax.plot(dates, reply_values, label="Replies")
    #ax.plot(dates)
    ax.plot(dates,total_values, label = "Aggregate")
    ax.set_xlabel("Date")
    ax.set_ylabel("(R+M)/M")
    #ax.set_title(f"{channelName} Activity")
    ax.set_title(f"{channelName} Aggregate Activity")

    # Set x-axis ticks and tick labels
    ax.xaxis.set_major_locator(MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(DateFormatter('%m/%d/%Y'))
    ax.tick_params(axis='x', labelrotation=45)
    #ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    plt.ylim(bottom=0, top=max(total_values)*1.1)
    ax.legend()

    savePath = f'/home/ravi/raviProject/DATA/1ExtractReplies/'

    plt.savefig(savePath+f"/{channelName}.png")
    
    
    with open(savePath+f'/{channelName}.json', 'w') as file:
        json.dump(messages, file)
    

    # break