#stock_price
import gradio as gr
import yfinance as yf
import matplotlib.pyplot as plt
import datetime
from io import BytesIO
from PIL import Image, ImageOps

#teachable_machine
from keras.models import load_model  # Kerasを動作させるためにはTensorFlowが必要です
import numpy as np

# 明確さのために科学的表記を無効にします
np.set_printoptions(suppress=True)

# モデルをロードします
model = load_model("teachable_machine/keras_model.h5", compile=False)
# ラベルをロードします
class_names = open("teachable_machine/labels.txt", "r").readlines()

def teachable_machine_predict(img):
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    image = img.convert('RGB')
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    prediction = model.predict(data)

    max_index = np.argmax(prediction) # 確率が一番高いインデクスを抽出

    #print(f"クラス {class_names[max_index][2:]} のスコア: {prediction[0][max_index] * 100}%")

    return max_index
    
# 75日前、25日前、5日前の日付を取得
dt_75 = (datetime.datetime.now() - datetime.timedelta(days=75)).strftime('%Y-%m-%d')
dt_25 = (datetime.datetime.now() - datetime.timedelta(days=25)).strftime('%Y-%m-%d')
dt_10 = (datetime.datetime.now() - datetime.timedelta(days=10)).strftime('%Y-%m-%d')

def get_stock_price(stock_symbol, start_date): 
    stock = yf.Ticker(stock_symbol) 
    data = stock.history(period="1d", start=start_date, end=None)
    return data["Close"] 

def plot_stock_price(stock_symbol, start_date): 
    prices = get_stock_price(stock_symbol, start_date)
    plt.figure(figsize=(8, 6))
    plt.plot(prices, color="red")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(f"Stock Price for {stock_symbol} since {start_date}")

    # グラフを画像に変換
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)

    return img

def main_stock_prices(stock_symbol):
    img_75 = plot_stock_price(stock_symbol, dt_75)
    img_25 = plot_stock_price(stock_symbol, dt_25)
    img_10 = plot_stock_price(stock_symbol, dt_10)

    predictation = []

    predictation.append(teachable_machine_predict(img_75))
    predictation.append(teachable_machine_predict(img_25))
    predictation.append(teachable_machine_predict(img_10))

    up = 0
    down = 0

    for num in predictation: 
        if num % 2 == 0:
            down += 1
        else:
            up += 1
    
    #print(f"up: {up}, down: {down}")

    predictation_result = "👍 上昇しそう" if up > down else "👎 下落しそう"

    return img_75, img_25, img_10, predictation_result

with gr.Blocks(theme='xiaobaiyuan/theme_brief') as iface:
    gr.Markdown(
        """ 
        # __シンプルストックフューチャー__
        📈 Teachable Machineを使って、株価の予測を行います。
        """
    )
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            stock_symbol = gr.Textbox(show_label=False, placeholder="証券コードを入力してください ")
            gr.Examples(["^GSPC","QQQ", "AAPL","BTC-USD","WLD-USD"], inputs=[stock_symbol], label="証券コード例 [yahoo finance]")
            submit_btn = gr.Button("予測する")
            output_predictation = gr.Label(label="予測結果")
        
        with gr.Column(scale=2):
            with gr.Tab("75 days"):
                output_75 = gr.Image(label="前75日間の株価")
            with gr.Tab("25 days"):
                output_25 = gr.Image(label="前25日間の株価")
            with gr.Tab("10 days"):
                output_10 = gr.Image(label="前10日間の株価")

    with gr.Row():
        gr.Markdown(
            """ 
            ## 📖 __使い方__
            1. 証券コードを入力してください。
            2. 予測するボタンをクリックしてください。
            3. 予測結果が表示されます。
            
            ## 🧬 __構造__
            #### Teachable Machine モデルのロード:
            Teachable Machine でトレーニングされたモデル (keras_model.h5) がロードされます。

            #### 株価データの取得と可視化:
            Yahoo Finance API を使用して、指定された証券コードに関する株価データを取得します。
            Matplotlib を使用して、取得した株価データを前75日、前25日、前10日の3つの期間にわたって可視化します。

            #### Teachable Machine による予測:
            取得した株価データの各期間について、Teachable Machine モデルを使用して上昇か下落かの予測を行います。

            #### Gradio インターフェース:
            Gradio を使用して、ユーザーが証券コードを入力し、予測ボタンをクリックすることで予測結果が表示されます。
            インターフェースは、入力欄、予測結果表示ラベル、および3つの画像タブで構成されています。
            
            ## ⚠️ __注意事項__
            - 予測結果はあくまで予測であり、実際の株価とは異なる場合があります。
            - このアプリケーションは教育目的で作成されています。
            - このアプリケーションは、株式市場の参考情報としてのみ使用することを目的としています。
            """
        )
    
    submit_btn.click(fn=main_stock_prices, inputs=[stock_symbol], outputs=[output_75, output_25, output_10, output_predictation])

iface.launch(share=True)