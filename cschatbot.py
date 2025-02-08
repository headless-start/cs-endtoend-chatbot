import os
import json
import random
import nltk
import ssl
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from flask import Flask, request, jsonify, render_template

# Set up SSL context and download NLTK data


ssl._create_default_https_context = ssl._create_unverified_context
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
    "tag": "cs2_skins",
    "patterns": [
        "How do CS2 skins work?",
        "What are some good CS2 skins?",
        "What affects the price of a skin?",
        "How can I trade skins?",
        "Are CS2 skins purely cosmetic?",
        "What are some rare and expensive CS2 skins?"
    ],
    "responses": [
        "CS2 skins are cosmetic weapon finishes that do not affect gameplay but can be bought, sold, and traded. Skins come in different rarities, conditions (float values), and some have unique features like StatTrak or rare patterns. They can be obtained from cases, operations, or by trading with other players.",
        
        "Some good and affordable skins include: AK-47 | Redline, M4A1-S | Nightmare, AWP | Atheris, Glock-18 | Water Elemental, and USP-S | Cortex. These skins have great designs and are budget-friendly for most players.",
        
        "The price of a skin is influenced by several factors, including: its rarity (Covert, Classified, Restricted, etc.), wear condition (Factory New to Battle-Scarred), demand, special features like StatTrak or Souvenir status, and rare pattern variations like the Blue Gem Karambit or Case Hardened AK-47.",
        
        "Rare and expensive skins include: AWP | Dragon Lore (one of the most famous and valuable skins), AK-47 | Case Hardened (especially with blue gem patterns), Karambit | Gamma Doppler Emerald, and M4A4 | Howl (which is no longer obtainable from cases). Some knives, like the Karambit and Butterfly Knife in rare finishes, can sell for thousands of dollars.",
        
        "You can trade skins using the Steam Community Market, direct trading with other players, or third-party marketplaces. Always be cautious of scams and double-check trade offers before accepting.",
        
        "Yes, CS2 skins are purely cosmetic and do not provide any in-game advantages. However, some skins are seen as status symbols due to their rarity and high price, making them desirable among collectors and traders."
    ]
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

vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)
tags, patterns = [], []

for intent in intents:
    for pattern in intent["patterns"]:
        tags.append(intent["tag"])
        patterns.append(pattern)

X = vectorizer.fit_transform(patterns)
y = tags
clf.fit(X, y)

# Define chatbot response function
def chatbot(input_text):
    input_text_transformed = vectorizer.transform([input_text])
    tag = clf.predict(input_text_transformed)[0]
    for intent in intents:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

# Create Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chatbot", methods=["POST"])
def chat():
    user_message = request.json["message"]
    bot_response = chatbot(user_message)
    return jsonify({"response": bot_response})

if __name__ == "__main__":
    app.run(debug=True)