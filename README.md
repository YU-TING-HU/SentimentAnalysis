跳至 [一、Sentiment Analysis of Hotel Reviews and Twitter Posts](#一sentiment-analysis-of-hotel-reviews-and-twitter-posts)

跳至 [二、Sentiment Analysis with Embeddings](#二sentiment-analysis-with-embeddings)

---

## 一、Sentiment Analysis of Hotel Reviews and Twitter Posts
## sentimentanalysis.py

程式中使用各種自然語言處理（NLP）套件（TextBlob、NLTK、EmoLex、VADER、BERT、MultinomialNB）對 Hotel Reviews 和 Twitter Posts 進行情感分析。

## 目錄

- [簡述](#簡述)
- [使用套件](#使用套件)
- [資料](#資料)
- [分析流程](#分析流程)

## 簡述

使用NLP相關方法來分析 Hotel Reviews 和 Twitter Posts 文字表達的情感並萃取關鍵詞。

## 使用套件

安裝以下套件：
- pandas
- google-colab
- seaborn
- matplotlib
- nltk
- textblob
- NRCLex
- vaderSentiment
- transformers
- torch
- scikit-learn

```bash
pip install pandas seaborn matplotlib nltk textblob NRCLex vaderSentiment transformers torch scikit-learn
```

## 資料

### Hotel Reviews 欄位
- `Index`
- `Name`
- `Area`
- `Review_Date`
- `Rating_attribute`
- `Review_Text`：顧客的評論
- `Rating(Out of 10)`：顧客給的評分

### Twitter Posts 欄位
- `textID`
- `text`
- `selected_text`
- `sentiment`

## 分析流程

### 1. 資料EDA、清整

- 資料的summary
- 缺失值和重複值

### 2. 使用 TextBlob 進行情感分析

- 分析 Reviews 的情感(數值)
- 計算評分和情感之間的皮爾森相關係數
- 視覺化皮爾森相關係數

### 3. 使用 TextBlob 和 NLTK 的關鍵詞分析

- 將 Reviews 的情感分類為正面、負面或中性
- 從正面和負面評論中整合出最常用的關鍵詞 (使用NLTK)

### 4. 使用 EmoLex 分類情緒

- 使用 EmoLex 分析 Reviews、Twitter 的情感
- 從 Reviews、Twitter 中篩選主要情緒(fear, anger, disgust, etc.)

### 5. 使用 VADER 進行情感分析

- 使用 VADER 分析 Twitter 的情感
- 將 Twitter 的情感分類為正面、負面或中性

### 6. 使用 BERT 進行情感預測

- 使用預訓練的 BERT 模型預測 Twitter 的情感

### 7. 使用 MultinomialNB 進行情感預測

- 使用 Twitter 訓練資料訓練 MultinomialNB 模型
- 使用訓練好的模型預測 Twitter 測試資料的情感

---

## 二、Sentiment Analysis with Embeddings
## SentimentAnalysis_Embedding.py

使用 sentence embeddings 和簡單神經網絡模型對 Twitter 資料進行情感分析。

## 目錄

- [使用套件](#套件)
- [分析流程](#程式流程)

## 套件

安裝以下套件：

- numpy
- pandas
- pickle
- seaborn
- scikit-learn
- torch
- sentence-transformers

```bash
pip install numpy pandas pickle seaborn scikit-learn torch sentence-transformers
```

## 程式流程

1. **Twitter資料**

```python
twitter_file = '/content/Tweets.csv'
df = pd.read_csv(twitter_file).dropna()
```

2. **將目標變項 y: 情感標籤轉為數值型**

```python
cat_id = {'neutral': 1, 'negative': 0, 'positive': 2}
df['class'] = df['sentiment'].map(cat_id)
```

3. **超參數**：

```python
BATCH_SIZE = 128
NUM_EPOCHS = 80
MAX_FEATURES = 10
```

4. **sentence embeddings 的預訓練模型**：

```python
emb_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v1')
X = emb_model.encode(df['text'].values)
```

5. **下載和載入pkl檔**：

```python
with open("/content/tweets_X.pkl", "wb") as output_file:
    pickle.dump(X, output_file)

with open("/content/tweets_X.pkl", "rb") as input_file:
    X = pickle.load(input_file)
```

6. **訓練集和測試集**：

```python
X_train, X_test, y_train, y_test = train_test_split(X, df['class'].values, test_size=0.5, random_state=123)
```

7. **定義 Dataset class**：

```python
class SentimentData(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = torch.Tensor(X)
        self.y = torch.Tensor(y).type(torch.LongTensor)
        self.len = len(self.X)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.X[index], self.y[index] 
```

8. **DataLoader**：

```python
train_ds = SentimentData(X= X_train, y = y_train)
test_ds = SentimentData(X_test, y_test)
train_loader = DataLoader(dataset=train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=15000)
```

9. **Model class**：

```python
class SentimentModel(nn.Module):
    def __init__(self, NUM_FEATURES, NUM_CLASSES, HIDDEN = 10):
        super().__init__()
        self.linear = nn.Linear(NUM_FEATURES, HIDDEN)
        self.linear2 = nn.Linear(HIDDEN, NUM_CLASSES)
        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.log_softmax(x)
        return x

model = SentimentModel(NUM_FEATURES = X_train.shape[1], NUM_CLASSES = 3)
```

10. **Train**：

```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters())
train_losses = []
for e in range(NUM_EPOCHS):
    curr_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred_log = model(X_batch)
        loss = criterion(y_pred_log, y_batch.long())

        curr_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_losses.append(curr_loss)
    print(f"Epoch {e}, Loss: {curr_loss}")
```

11. **Loss plot**：

```python
sns.lineplot(x=list(range(len(train_losses))), y= train_losses)
```

12. **Test**：

```python
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        y_test_pred_log = model(X_batch)
        y_test_pred = torch.argmax(y_test_pred_log, dim = 1)

y_test_pred_np = y_test_pred.squeeze().cpu().numpy()
acc = accuracy_score(y_pred=y_test_pred_np, y_true = y_test)
print(f"The accuracy of the model is {np.round(acc, 3)*100}%.")

most_common_cnt = Counter(y_test).most_common()[0][1]
print(f"Naive Classifier: {np.round(most_common_cnt / len(y_test) * 100, 1)} %")
```


