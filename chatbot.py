import pandas as pd 
import spacy 
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC 
from sklearn.pipeline import make_pipeline 
from sklearn.feature_extraction.text import TfidfVectorizer 
import torch # type: ignore
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup 
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler 
import os
import sqlite3
import sqlite3
import re
import logging
# Load the CSV file

csv_file = r"C:\Users\Lenovo\Downloads\intents_and_exampless.csv"
df_intents = pd.read_csv(csv_file)

# Display the first few rows to understand its structure
print(df_intents.head())


# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Encode labels
label_encoder = LabelEncoder()
df_intents['Intent'] = label_encoder.fit_transform(df_intents['Intent'])

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df_intents['Example'], df_intents['Intent'], test_size=0.2, random_state=42)

# Create a text classification pipeline
pipeline = make_pipeline(TfidfVectorizer(), SVC(probability=True))

# Train the model
pipeline.fit(X_train, y_train)

# Test the model
accuracy = pipeline.score(X_test, y_test)
print(f"Model accuracy: {accuracy * 100:.2f}%")


# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def encode_data(texts, labels, tokenizer, max_length=128):
    input_ids = []
    attention_masks = []
    for text in texts:
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    return input_ids, attention_masks, labels

train_inputs, train_masks, train_labels = encode_data(X_train.tolist(), y_train.tolist(), tokenizer)
test_inputs, test_masks, test_labels = encode_data(X_test.tolist(), y_test.tolist(), tokenizer)

# Create DataLoader

batch_size = 4

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

# Load BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder.classes_))
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
total_steps = len(train_dataloader) * 4  # Number of epochs

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# # Training function
# def train(model, dataloader, optimizer, scheduler, device):
#     model.train()
#     total_loss = 0
#     for batch in dataloader:
#         batch_inputs, batch_masks, batch_labels = [item.to(device) for item in batch]
#         model.zero_grad()
#         outputs = model(batch_inputs, token_type_ids=None, attention_mask=batch_masks, labels=batch_labels)
#         loss = outputs.loss
#         total_loss += loss.item()
#         loss.backward()
#         optimizer.step()
#         scheduler.step()
#     return total_loss / len(dataloader)

# # Training loop
# epochs = 3
# for epoch in range(epochs):
#     loss = train(model, train_dataloader, optimizer, scheduler, device)
#     print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")

# # Evaluation function

tokenizer = BertTokenizer.from_pretrained(r"C:\Users\Lenovo\Downloads\Model-20240704T051704Z-001\Model")
# Load the model
model = BertForSequenceClassification.from_pretrained(r"C:\Users\Lenovo\Downloads\Model-20240704T051704Z-001\Model")

def evaluate(model, dataloader, device):
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    for batch in dataloader:
        batch_inputs, batch_masks, batch_labels = [item.to(device) for item in batch]
        with torch.no_grad():
            outputs = model(batch_inputs, token_type_ids=None, attention_mask=batch_masks, labels=batch_labels)
            loss = outputs.loss
            logits = outputs.logits
            total_eval_loss += loss.item()
            preds = torch.argmax(logits, dim=1).flatten()
            accuracy = (preds == batch_labels).cpu().numpy().mean() * 100
            total_eval_accuracy += accuracy
    return total_eval_loss / len(dataloader), total_eval_accuracy / len(dataloader)

# Evaluate the model
loss, accuracy = evaluate(model, test_dataloader, device)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.2f}%")

# model.save_pretrained(r"C:\Users\Lenovo\Desktop\Chatbot")
# tokenizer.save_pretrained(r"C:\Users\Lenovo\Desktop\Chatbot")


# Load the new dataset
new_csv_file = r"C:\Users\Lenovo\Downloads\test-customer-dataset.csv"
df_customers = pd.read_csv(new_csv_file)

# Display the first few rows to understand its structure
print(df_customers.head())


# Read the CSV file
csv_file = r"C:\Users\Lenovo\Downloads\test-customer-dataset.csv"
df = pd.read_csv(csv_file)


# Create a connection to the SQLite database
conn = sqlite3.connect('chatbot.db')
cursor = conn.cursor()

# Convert the DataFrame to a SQL table
df_customers.to_sql('customers', conn, if_exists='replace', index=False)

# Verify by querying the database
cursor.execute("SELECT * FROM customers LIMIT 5")
rows = cursor.fetchall()
for row in rows:
    print(row)

# Commit and close the connection
conn.commit()
conn.close()

import sqlite3

def fetch_customer_balance(Surname):
    conn = sqlite3.connect('chatbot.db')
    c = conn.cursor()
    c.execute('SELECT Balance FROM customers WHERE Surname = ?', (Surname,))
    result = c.fetchone()
    conn.close()
    if result:
        return f"Your current balance is ${result[0]}."
    else:
        return "No balance data available."

def transfer_money(name, receiver_name, amount):
    conn = sqlite3.connect('chatbot.db')
    c = conn.cursor()

    # Fetch sender's balance by surname
    c.execute('SELECT Balance FROM customers WHERE Surname = ?', (name,))
    sender_balance = c.fetchone()

    # Fetch receiver's balance by surname
    c.execute('SELECT Balance FROM customers WHERE Surname = ?', (receiver_name,))
    receiver_balance = c.fetchone()

    if not receiver_balance:
        conn.close()
        return f"Receiver with Name {receiver_name} not found."
    if not sender_balance or sender_balance[0] < amount:
        conn.close()
        return f"Insufficient funds."

    try:
        # Begin transaction
        print(f"Sender's balance before transfer: ${sender_balance[0]}")
        print(f"Receiver's balance before transfer: ${receiver_balance[0]}")
        conn.execute('BEGIN TRANSACTION')

        # Decrease balance of sender
        c.execute('UPDATE customers SET Balance = Balance - ? WHERE Surname = ?', (amount, name))

        # Increase balance of receiver
        c.execute('UPDATE customers SET Balance = Balance + ? WHERE Surname = ?', (amount, receiver_name))

        # Commit transaction
        conn.commit()

        c.execute('SELECT Balance FROM customers WHERE Surname = ?', (name,))
        updated_sender_balance = c.fetchone()

        c.execute('SELECT Balance FROM customers WHERE Surname = ?', (receiver_name,))
        updated_receiver_balance = c.fetchone()

        # Print balances after the transaction
        print(f"Sender's balance after transfer: ${updated_sender_balance[0]}")
        print(f"Receiver's balance after transfer: ${updated_receiver_balance[0]}")


        conn.close()
        return f"Successfully transferred ${amount} to {receiver_name}."
    except sqlite3.Error as e:
        # Rollback transaction in case of error
        conn.rollback()
        conn.close()
        return f"Failed to transfer money due to: {str(e)}"
    
    import re
session_data = {}
def classify_intent(query):
      inputs = tokenizer.encode_plus(
        query,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )


      input_ids = inputs['input_ids'].to(device)
      attention_mask = inputs['attention_mask'].to(device)

      with torch.no_grad():
          outputs = model(input_ids, token_type_ids=None, attention_mask=attention_mask)

      logits = outputs.logits
      intent = torch.argmax(logits, dim=1).cpu().numpy()[0]

      classified_intent = label_encoder.inverse_transform([intent])[0]

    # Print statement to check the classified intent
      print(f"Query: '{query}' classified as Intent: '{classified_intent}'")

      return classified_intent

# def extract_entities(text):
#     doc = nlp(text)
#     entities = {ent.label_: ent.text for ent in doc.ents}

#     # Use regex to capture names if not recognized by SpaCy
#     if "PERSON" not in entities:
#         # Look for a pattern like "to [Name]"
#         match = re.search(r'\bto\s+([A-Z][a-z]+)\b', text)
#         if match:
#             entities["PERSON"] = match.group(1)
#     return entities
def extract_entities(text):
    doc = nlp(text)
    entities = {ent.label_: ent.text for ent in doc.ents}

    # Use regex to capture names if not recognized by SpaCy
    if "PERSON" not in entities:
        # Look for patterns like "to [Name]", "Send [Name]", or "pay [Name]"
        match = re.search(r'\b(?:to|send|pay)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\b', text, re.IGNORECASE)
        if match:
            entities["PERSON"] = match.group(1)
    return entities


def generate_response(user_input, name):
    global session_data
    intent = classify_intent(user_input)
    entities = extract_entities(user_input)

    # Update session data with new entities
    session_data.update(entities)

    if intent == "check_balance":
        return fetch_customer_balance(name)
    elif intent == "transfer_money":
        amount = session_data.get("MONEY", None)
        receiver_name = session_data.get("PERSON", None)

        if amount and receiver_name:
            # Clear the session data after use
            session_data.clear()
            amount = float(amount.replace('$', '').replace(',', ''))
            return transfer_money(name, receiver_name, amount)
        else:
            return "Please provide receiver's name and amount to transfer."
    elif intent == "check_human":
        return "I am an AI created to assist you with financial inquiries."
    elif intent == 'open_account':
        return "To open a new account, please visit our website or nearest branch with your identification documents."
    elif intent == 'close_account':
        return "To close your account, please visit our nearest branch or contact our customer support."
    elif intent == "loan_inquiry":
        return "You can inquire about loans and their eligibility criteria on our website or by visiting our branch."
    elif intent == 'credit_card_application':
        return "You can apply for a credit card through our website or by visiting our nearest branch."
    elif intent == 'contact_support':
        return "You can contact our customer support through the chat feature on our website or by calling our support number."
    else:
        return "Sorry, I don't understand your query."

def chat():
    print("Welcome to the Financial Chatbot! How can I assist you today?")
    name = input("Please enter your name: ")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit"]:
            print("Chatbot: Thank you for using the Financial Chatbot. Have a great day!")
            break
        response = generate_response(user_input, name)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    chat()