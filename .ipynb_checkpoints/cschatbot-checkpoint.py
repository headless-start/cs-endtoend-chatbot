import os
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Set up SSL context and download NLTK data
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Define your intents
intents = [
    {
        "tag": "greeting",
        "patterns": ["Hi", "Hello", "Hey", "What's up", "Yo"],
        "responses": ["Hey there, ready to frag?", "Hello, need some CS strats?", "Yo, let's talk Counter-Strike!", "Hey! How's your aim today?"]
    },
    {
        "tag": "goodbye",
        "patterns": ["Bye", "See you later", "Goodbye", "GG", "Take care"],
        "responses": ["Goodbye, see you on the server!", "GG, well played!", "Take care, and keep practicing!", "Later! Don't forget to check your angles!"]
    },
    {
        "tag": "thanks",
        "patterns": ["Thank you", "Thanks", "Thanks a lot", "I appreciate it"],
        "responses": ["No problem! Hope that helps your game.", "You're welcome, keep grinding!", "Glad I could help. Now go clutch!"]
    },
    {
        "tag": "about",
        "patterns": ["What can you do", "Who are you", "What are you", "What is your purpose"],
        "responses": ["I'm your CS:GO strat bot!", "I help players improve at Counter-Strike.", "Need strats, economy tips, or movement tricks? I'm here to help!", "I provide tips and strategies for Counter-Strike."]
    },
    {
        "tag": "help",
        "patterns": ["Help", "I need help", "Can you help me", "What should I do"],
        "responses": ["Sure, what do you need help with? Strats, aiming, or economy?", "I'm here to help! Need a good T-side strategy?", "Want to know how to clutch better? Ask away!"]
    },
    {
        "tag": "cs_strats",
        "patterns": ["What are good CS strategies?", "Best T-side strategies?", "Best CT-side strategies?", "How do I execute a site take?", "What are some good strats for Dust 2?"],
        "responses": [
            "A good T-side strategy is the 'default': Spread across the map, gain control, and look for picks before executing a site.",
            "For CT-side, holding crossfires and playing off your teammate's information is key.",
            "On Dust 2, you can try a long A split with 3 players taking long control while 2 lurk mid or short.",
            "Executing a site take? Use utility! Smokes for vision denial, flashes to blind enemies, and molotovs to clear corners."
        ]
    },
    {
        "tag": "cs_economy",
        "patterns": ["How does the CS economy work?", "When should I buy?", "What is an eco round?", "How do I manage my money in CS?"],
        "responses": [
            "CS economy is all about managing your team's money to maximize gun rounds. Losing rounds gives you money, but not enough to buy every round.",
            "Eco rounds are when you save money by buying only pistols or nothing at all to afford rifles in the next round.",
            "A full buy usually means an AK/M4, armor, and utility. If you don't have enough for that, consider half-buying or saving."
        ]
    },
    {
        "tag": "cs_weapons",
        "patterns": ["What is the best weapon in CS?", "Should I use M4A4 or M4A1-S?", "What pistols are best for eco rounds?", "Is the AWP worth buying?"],
        "responses": [
            "It depends! The AK-47 is the best rifle on T-side due to one-shot headshots. M4A4 is better for spraying, while M4A1-S is more accurate with less ammo.",
            "For eco rounds, the P250 and Deagle are strong choices due to their high damage.",
            "The AWP is powerful but expensive. If your team needs impact and you can hit shots, it's worth buying!"
        ]
    },
    {
        "tag": "cs_movement",
        "patterns": ["How do I improve my movement in CS?", "What is counter-strafing?", "How can I bhop?", "How do I peek properly?"],
        "responses": [
            "Counter-strafing is stopping your movement before shooting to increase accuracy. Tap the opposite key before firing!",
            "Bunny hopping (bhopping) requires perfect timing on your jumps while strafing in the air.",
            "For peeking, use jiggle peeks to get info, wide swings to catch enemies off guard, and shoulder peeks to bait shots."
        ]
    },
    {
        "tag": "cs_ranks",
        "patterns": ["How do I rank up in CS?", "What is the CS ranking system?", "How do I get better at matchmaking?", "What are the ranks in CS?"],
        "responses": [
            "Ranking up in CS requires consistent wins and good performance. Playing with a team and communicating well can help.",
            "CS ranks range from Silver to Global Elite in CS:GO. In CS2, the ranking system is different, using skill groups per map.",
            "To get better at matchmaking, focus on game sense, utility usage, and crosshair placement."
        ]
    }
]

# Set up the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data: prepare patterns and corresponding tags
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Train the model
x = vectorizer.fit_transform(patterns)
y = tags  # Correct assignment of y
clf.fit(x, y)

# Define the chatbot function
def chatbot(input_text):
    input_text_transformed = vectorizer.transform([input_text])
    tag = clf.predict(input_text_transformed)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response

# Streamlit setup for the chat interface
counter = 0

def main():
    global counter
    st.title("Chatbot")
    st.write("Welcome to the chatbot. Please type a message and press Enter to start the conversation.")

    counter += 1
    user_input = st.text_input("You:", key=f"user_input_{counter}")

    if user_input:
        response = chatbot(user_input)
        st.text_area("Chatbot:", value=response, height=100, max_chars=None, key=f"chatbot_response_{counter}")

        if response.lower() in ['goodbye', 'bye']:
            st.write("Thank you for chatting with me. Have a great day!")
            st.stop()

if __name__ == '__main__':
    main()
