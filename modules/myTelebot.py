import requests

def telegram_bot_sendtext(bot_message):

   bot_token = '5732485928:AAHSXwkZpMXD_Hu2fuHvsZd5R2jiIAUnVu4'
   bot_chatID = '1246458300'
   send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text=' + bot_message

   response = requests.get(send_text)

   return response.json()
