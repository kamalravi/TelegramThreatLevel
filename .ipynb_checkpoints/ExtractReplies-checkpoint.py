import pandas as pd
import numpy as np

chats = ['absoluteTruth1776', 'AlexJones', 'AlexjonesInfoWars', 'DonaldTrumpJr', 'FreedomFighters',
         'InfoWars.com', 'PrayingMedic', 'RTM', 'ThePatriotVoice','TheTrumpRepublicans', 
         'TrumpSupportersChannel', 'WeTheMedia', 'WhiteLivesMatter']
# chats = ['InfoWars.com']

for idx, name in enumerate(chats):
    
    filePath = f'/home/ravi/UCF Dropbox/KAMALAKKANNAN RAVI/guyonDesktop/DATA_AutomatedHarmDetection/allChats/{name}/{name}_LinkAndPlot.json'
    
    print(idx+1,filePath)
    
    data = pd.read_json(filePath, orient='records')
    data = pd.DataFrame.transpose(data)
    print(data.shape)
    data.reset_index(drop=True, inplace=True)
    # print(data.head(1))
    
    channelDF = pd.DataFrame()
    for idx, msgItem in data.iterrows():
        
        # print(idx, 'msgOrder')
        # print(msgItem.shape)
        # print(msgItem)
        # print(len(msgItem['text']))
        
        msg = []
        if isinstance(msgItem['text'], str):
            msg = msgItem['text']
        else:
            for msgText in msgItem['text']:
                if isinstance(msgText, str):
                    msg.append(msgText)
                elif len(msgText) > 0:
                    # print(len(msgText))
                    # print(msgText['text'])
                    msg.append(msgText['text'])
                
        msg = ' '.join(msg)
        msgDate=msgItem['date_unixtime']
        
        replies = []
        repliesDate = [] 
        for reply in msgItem['replies']:
            if len(reply) > 0:
                # print(type(reply))
                # print(reply['text'])
                replies.append(reply['text'])
                repliesDate.append(reply['date_unixtime'])
            
        msg = [msg for _ in range(len(replies))]
        msgDate = [msgDate for _ in range(len(replies))]

        # print(len(msgDate), len(msg), len(repliesDate), len(replies))
        tempDF = {'msgDate': msgDate, 'msg': msg, 'replyDate': repliesDate, 'reply': replies}
        tempDF = pd.DataFrame(tempDF)

        channelDF = pd.concat([channelDF, tempDF], ignore_index=True)  
        # print(channelDF.shape)
    channelDF['telegramChannel']=name
    print(channelDF.shape)
    
    write_path=f'/home/ravi/UCF Dropbox/KAMALAKKANNAN RAVI/guyonDesktop/DATA_AutomatedHarmDetection/allChatsReplies/{name}_Date_Replies.json'
    channelDF.to_json(write_path, orient='records')
        # break
    # break