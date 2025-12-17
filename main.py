from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from typing import List, Optional

# model_functions.py からモデルロードと前処理関数をインポート
# 注: .venv環境で起動しているため、相対インポート（.model_functions）を使います
from model_functions import LGBM_MODEL, preprocess_data

# FastAPIインスタンスの作成
app = FastAPI(
    title="銀行顧客ターゲティング予測API",
    description="LightGBMモデルを使用して、顧客が定期預金に申し込むかを予測します。"
)

# 💡 データを受け取るためのPydanticモデルの定義
# ⚠️ あなたのデータセットに合わせて、すべての特徴量を正確に定義してください！
class CustomerData(BaseModel):
    # 例: 銀行データセットの特徴量 (必ず確認して修正してください)
    age: int
    job: str
    marital: str
    education: str
    default: str
    balance: float
    housing: str
    loan: str
    contact: str
    day: int
    month: str
    duration: int
    campaign: int
    pdays: int
    previous: int
    poutcome: str
    # ⚠️ ここにモデル学習に使ったすべてのカラムを定義してください！


# ヘルスチェックエンドポイント（APIが生きているか確認用）
@app.get("/")
def health_check():
    return {"status": "ok", "model_version": "LGBM v1.0"}

# 予測エンドポイント
@app.post("/predict")
def predict(data_list: List[CustomerData]):
    
    # PydanticモデルのリストをPandas DataFrameに変換
    # .model_dump() はPydantic V2以降の標準的な辞書変換メソッド
    data_df = pd.DataFrame([data.model_dump() for data in data_list])
    
    # 1. データの事前処理
    processed_data = preprocess_data(data_df)
    
    # 2. 予測の実行
    if LGBM_MODEL is None:
        # モデルがロードされていない場合はエラーを返す
        return {"error": "Model not loaded."}, 500

    # LightGBMは予測確率を返す。[:, 1]でクラス1（申し込む）の確率を取得
    predictions = LGBM_MODEL.predict_proba(processed_data)[:, 1] 

    # 3. 結果の整形
    results = [
        {"probability_subscribe": float(prob), "prediction": int(prob > 0.5)}
        for prob in predictions
    ]
    
    return results