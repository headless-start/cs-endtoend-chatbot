# Counter-Strike 2 Chatbot  

## 📌 Project Overview  
This project is an end-to-end **Counter-Strike 2 Chatbot** that provides real-time answers to user queries about the game. Built entirely using **Flask**, the chatbot allows users to ask questions about game economy, map strategies, skins, and more.  

The backend leverages **TF-IDF vectorization** and **Logistic Regression** for intent classification, ensuring precise and relevant responses.  

---

## 🚀 Key Features  
1. **Flask-Powered Chat Interface**:
   - The entire chatbot runs on Flask, handling both frontend and backend.  
   - Users can type questions and receive instant responses.  
   - Supports queries about game economy, map strategies, skins, and general gameplay tips.    
2. **Intent Classification with Machine Learning**:  
   - Utilizes **TF-IDF (Term Frequency-Inverse Document Frequency)** for feature extraction.  
   - **Logistic Regression** model classifies user queries into predefined categories.  
   - Ensures fast and accurate response generation.  

---

## 🔍 How It Works  
1. **User Input**:  
   - Users enter their queries via the Flask-based chat interface (e.g., "What is the best economy strategy for Mirage?").  
2. **Backend Processing**:  
   - The input text is processed using **TF-IDF vectorization**.  
   - The **Logistic Regression model** predicts the intent of the query.  
   - A relevant response is retrieved from the predefined knowledge base.  
3. **Response Generation**:  
   - Flask sends the generated response back to the user via the chat interface.  

---

## 🛠 System Requirements  
### Dependencies  
- Python 3.8+  
- Required Libraries:  
  ```bash
  pip install flask nltk pandas scikit-learn
  
---

## 📄 License  
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
