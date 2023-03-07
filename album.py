# 必要なモジュールのインポート
import torch
from abeke import transform, Net
from flask import Flask, request, render_template, redirect
import io
from PIL import Image
import base64

# 学習済みモデルをもとに推論を行う
def predict(img):
    # ネットワークの準備
    net = Net().cpu().eval()
    # 学習済みモデルの重み（face_res18.pt）を読み込むt
    # net.load_state_dict(torch.load('code/face_res18.pt', map_location=torch.device('cpu')))
    # Renderへデプロイする際は、'./face_res18.pt'とする。
    net.load_state_dict(torch.load('./face_res18.pt', map_location=torch.device('cpu')))
    # データの前処理
    img = transform(img)
    img = img.unsqueeze(0) # 1次元増やす
    # 推論
    y = torch.argmax(net(img), dim=1).cpu().detach().numpy()
    return y

# 推論したラベルから人物を判定
def getName(label):
    if label == 0:
        return '父'
    elif label == 1:
        return '母'
    elif label == 2:
        return '長男'
    elif label == 3:
        return '長女'

# Flaskのインスタンス化
app = Flask(__name__)

# アップロードされる拡張子の制限
# pngファイルはimage = Image.open(file).convert('RGB')でconvertしないとエラーとなる
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif', 'jpeg'])

# 拡張子が適切かどうかをチェック
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# URLにアクセスがあった場合の挙動の設定
@app.route('/', methods = ['GET', 'POST'])
def predicts():
    # POSTメソッドの定義
    if request.method == 'POST':
        # ファイルがなかった場合の処理
        if 'filename' not in request.files:
            return redirect(request.url)
        # データの取り出し
        file = request.files['filename']
        # ファイルのチェック
        if file and allowed_file(file.filename):
            # 画像ファイルに実行される処理
            # 画像読み込み用バッファを確保
            buf = io.BytesIO()
            image = Image.open(file).convert('RGB')
            # image = Image.open(file)
            # 画像データをバッファに書き込む
            image.save(buf, 'png')
            # バイナリデータをbase64でエンコードしてutf-8でデコード
            base64_str = base64.b64encode(buf.getvalue()).decode('utf-8')
            base64_data = 'data:image/png;base64,{}'.format(base64_str)

            # 入力された画像に対して推論
            pred = predict(image)
            familyName_ = getName(pred)
            return render_template('result.html', familyName=familyName_, image=base64_data)
        return redirect(request.url)

    # GETメソッドの定義
    elif request.method == 'GET':
        return render_template('index.html')

# アプリケーションの実行の定義
if __name__ == '__main__':
    app.run(debug=True)
