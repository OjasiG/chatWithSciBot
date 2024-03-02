css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #2b313e
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #ffcbd1;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <image src = "https://as2.ftcdn.net/v2/jpg/02/10/96/95/1000_F_210969565_cIHkcrIzRpWNZzq8eaQnYotG4pkHh0P9.jpg" style="height: 70px; width: 78px; object-fit: fill;" >
    </div>
    <div class="message">{{MSG}}</div>
</div>'''


user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://as1.ftcdn.net/v2/jpg/05/98/48/88/1000_F_598488869_fiLUgajDxyaoxE9D3SuHMZfD56IjrBXe.jpg" style="height: 70px;width: 78px;object-fit: fill;">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
