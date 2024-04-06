# 📈Simple_Stock_Future
 Teachable Machineを使って、株価の予測を行うアプリケーションです。   

### アプリケーションURL 
 こちらのアプリケーションは Hugging Face というクラウド実行サービスを利用しています。
 https://huggingface.co/spaces/Aquly/simple_stock_future

## 🧬構造
 1. `App/generate_graph.py` から様々な株価のチャートパターンを出力し、**Teachable Machine** でモデルをトレーニングします。
 2. トレーニングしたモデルを `App/stock_price.py`でロードして利用可能にします。
 3. Yahoo Finance API を使用して、指定された証券コードに関する株価データを取得します。
 4. Matplotlib を使用して、取得した株価データを前75日、前25日、前10日の3つの期間にわたって可視化します。
 5. 取得した株価データの各期間について、ロードしたモデルを使用して上昇か下落かの予測を行います。

## ⚠️ 注意事項
- 予測結果はあくまで予測であり、実際の株価とは大きく異なる場合がございます。
- このアプリケーションは教育目的で作成されています。
