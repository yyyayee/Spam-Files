import re
import pythainlp
import pandas as pd
from sklearn import metrics
from pythainlp import word_tokenize
from pythainlp.corpus import thai_stopwords
from sklearn.naive_bayes import MultinomialNB
from flask import Flask, request, render_template
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)
df = pd.read_excel('SpamDataset.xlsx')
stop_words = list(thai_stopwords())

def clean_thai_text(text):
    words = word_tokenize(text, engine='longest', keep_whitespace=False)
    words = [word for word in words if word not in stop_words]
    cleaned_words = []
    for word in words:
        cleaned_word = re.sub(r'[^ก-๙]', '', word)
        if cleaned_word:
            cleaned_words.append(cleaned_word)
    return ' '.join(cleaned_words)

df['v2_cleaned'] = df['v2'].apply(clean_thai_text)
df.columns = ['label', 'msg', 'clean_msg']
df['label_sign'] = df.label.map({'ham': 0, 'spam': 1})

X_train, X_test, y_train, y_test = train_test_split(df['clean_msg'], df['label_sign'], test_size=0.2, random_state=42)

vect = CountVectorizer()
vect.fit(X_train)
X_train_dim = vect.transform(X_train)
X_test_dim = vect.transform(X_test)

clf_mnb = MultinomialNB(alpha=0.2)
clf_mnb.fit(X_train_dim, y_train)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        new_message = request.form['message']
        cleaned_message = clean_thai_text(new_message)
        new_message_vectorized = vect.transform([cleaned_message])
        prediction = clf_mnb.predict(new_message_vectorized)[0]
        probability = clf_mnb.predict_proba(new_message_vectorized)[0][1] * 100  # Spam probability

        if prediction == 1:
            result = f"ข้อความนี้อาจเป็น Spam ({probability:.2f}%)"
        else:
            result = f"ข้อความนี้เป็น Ham ({100 - probability:.2f}%)"

        return render_template('index.html', result=result)
    
    return render_template('index.html', result='')

if __name__ == '__main__':
    app.run(debug=True)