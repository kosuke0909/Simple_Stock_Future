from keras.models import load_model  # Kerasを動作させるためにはTensorFlowが必要です
from PIL import Image, ImageOps  # PILの代わりにpillowをインストールしてください
import numpy as np

# 明確さのために科学的表記を無効にします
np.set_printoptions(suppress=True)

# モデルをロードします
model = load_model("teachable_machine/keras_model.h5", compile=False)

# ラベルをロードします
class_names = open("teachable_machine/labels.txt", "r").readlines()

# kerasモデルにフィードするための正しい形状の配列を作成します
# 配列に入れることができる画像の'長さ'または数は、
# この場合、形状タプルの最初の位置によって決定されます
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# これをあなたの画像へのパスに置き換えてください
image = Image.open("Datasets/stock_img/^GSPC5.png").convert("RGB")

# 画像を少なくとも224x224にリサイズし、中央からクロップします
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# 画像をnumpy配列に変換します
image_array = np.asarray(image)

# 画像を正規化します
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# 画像を配列にロードします
data[0] = normalized_image_array

# モデルを予測します
prediction = model.predict(data)

# 全クラスのスコアを出力します
for i, score in enumerate(prediction[0]):
    print(f"クラス {class_names[i][2:]} のスコア: {score * 100}%")

# 最も高いスコアのクラスを出力します
print(f"\nこの画像は {class_names[np.argmax(prediction[0])][2:]} です。")