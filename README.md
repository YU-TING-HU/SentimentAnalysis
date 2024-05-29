## [一、Sentiment Analysis of Hotel Reviews and Twitter Posts](#sentimentanalysis.py)
## [二、](#SentimentAnalysis_Embedding.py)
---

## 一、Sentiment Analysis of Hotel Reviews and Twitter Posts
### sentimentanalysis.py

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

## 二、
### SentimentAnalysis_Embedding.py

