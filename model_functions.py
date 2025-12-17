import joblib
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from typing import Optional, Any, List
import numpy as np

# --- 1. モデルのロード ---
# 型ヒントを Any にすることで、BoosterかClassifierかを問わず柔軟に扱えるようにします
LGBM_MODEL: Any = None 
# 学習時の項目名を保存する変数
EXPECTED_FEATURES: List[str] = []

try:
    LGBM_MODEL = joblib.load('lgbm_model.pkl')
    print("✅ モデルファイル (lgbm_model.pkl) のロードに成功しました。")
    
    # AIモデルが学習時に使った「正しい項目の名前と順番」を取得します
    if hasattr(LGBM_MODEL, 'feature_name_'):
        EXPECTED_FEATURES = LGBM_MODEL.feature_name_
    elif hasattr(LGBM_MODEL, 'feature_name'):
        # Boosterオブジェクトの場合
        EXPECTED_FEATURES = LGBM_MODEL.feature_name()
    
    if EXPECTED_FEATURES:
        print(f"📋 AIが期待している項目 ({len(EXPECTED_FEATURES)}個): {EXPECTED_FEATURES}")
    else:
        print("⚠️ モデルから項目名を取得できませんでした。")

except FileNotFoundError:
    print("🚨エラー: 'lgbm_model.pkl' が見つかりません。")

# --- 2. 前処理関数 ---
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    入力されたデータをAIが予測できる形に整え、
    さらに項目数の不足（16個 vs 17個）を自動で解消します。
    """
    df_processed = df.copy()
    
    # カテゴリ変数のリスト
    categorical_cols = [
        'job', 'marital', 'education', 'default', 'housing', 
        'loan', 'contact', 'month', 'poutcome'
    ]
    
    # --- Label Encodingの適用 ---
    for col in categorical_cols:
        if col in df_processed.columns:
            le = LabelEncoder()
            # 文字列に変換して欠損値を埋め、数値化
            df_processed[col] = le.fit_transform(df_processed[col].astype(str).fillna('unknown'))
            
    # pdays の処理 (-1 を 99999 に置き換える)
    if 'pdays' in df_processed.columns:
        df_processed['pdays'] = df_processed['pdays'].replace(-1, 99999)

    # --- 🚨 最重要：項目の過不足調整 (16個を17個にする) ---
    if EXPECTED_FEATURES:
        # 1. モデルが期待しているのに、今のデータに存在しない項目を探して 0 で埋める
        for col_name in EXPECTED_FEATURES:
            if col_name not in df_processed.columns:
                df_processed[col_name] = 0  # 足りない項目（例：idなど）を0で作成
        
        # 2. モデルが学習した時と「全く同じ並び順」に列を並び替える
        # これをやらないと、数値が別の項目として判定されてしまいます
        df_processed = df_processed[EXPECTED_FEATURES]

    return df_processed