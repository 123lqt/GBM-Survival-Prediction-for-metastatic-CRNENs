import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import warnings
import sys

# 忽略常见的无关警告
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# 兼容某些环境里已弃用的 np.bool
if not hasattr(np, 'bool'):
    np.bool = bool

# Page config
st.set_page_config(
    page_title="GBM Survival Prediction (Cox GBM)",
    page_icon="📈",
    layout="wide"
)

# 生存预测时间点（假设数据 Time 单位为“月”）
PRED_YEARS = [1, 3, 5]
YEAR_TO_MONTHS = {1: 12, 3: 36, 5: 60}

# 当前模型特征（与训练一致）
DEFAULT_FEATURES = [
    "Age",
    "Sex",
    "Histologic",
    "N_stage",
    "Grade",
    "Bone",
    "Liver",
    "Tumor_Size",
    "Primary_Site_Surgery",
    "Chemotherapy",
]

# Feature labels (English)
FEATURE_LABELS_CN = {
    "Age": "Age group",
    "Sex": "Sex",
    "Histologic": "Histologic type (ICD-O morphology code)",
    "N_stage": "N stage",
    "Grade": "Pathologic grade",
    "Bone": "Bone metastasis",
    "Liver": "Liver metastasis",
    "Tumor_Size": "Tumor size",
    "Primary_Site_Surgery": "Primary site surgery",
    "Chemotherapy": "Chemotherapy",
}

# Encoded value → English text mapping
OPTIONS = {
    "Age": [(0, "<65 years"), (1, "≥65 years")],
    "Sex": [(0, "Female"), (1, "Male")],
    "Grade": [(1, "I"), (2, "II"), (3, "III"), (4, "IV"), (5, "Unknown")],
    "Chemotherapy": [(0, "No"), (1, "Yes")],
    "N_stage": [(0, "N0"), (1, "N1")],
    "Tumor_Size": [(1, "≤2 cm"), (2, "2–4 cm"), (3, ">4 cm")],
    "Bone": [(0, "No"), (1, "Yes")],
    "Liver": [(0, "No"), (1, "Yes")],
    "Histologic": [
        (0, "8013"), (1, "8041"), (2, "8240"), (3, "8244"),
        (4, "8245"), (5, "8246"), (6, "8249"), (7, "8510"),
    ],
    # If this mapping does not match your data, please tell me: default 0=No, 1=Yes
    "Primary_Site_Surgery": [(0, "No"), (1, "Yes")],
}

def _select_option(label_cn: str, opts: list, key: str):
    labels = [lab for _, lab in opts]
    sel = st.selectbox(label_cn, labels, key=key)
    code = [code for code, lab in opts if lab == sel][0]
    return code


# Load model payload (contains model and features); provide numpy._core fallback for some envs
@st.cache_resource
def load_model_payload():
    model_path = 'gbm_cox_model.pkl'
    try:
        payload = joblib.load(model_path)
    except ModuleNotFoundError as e:
        if 'numpy._core' in str(e):
            import numpy as _np
            sys.modules['numpy._core'] = _np.core
            sys.modules['numpy._core._multiarray_umath'] = _np.core._multiarray_umath
            sys.modules['numpy._core.multiarray'] = _np.core.multiarray
            sys.modules['numpy._core.umath'] = _np.core.umath
            payload = joblib.load(model_path)
        else:
            raise
    return payload

@st.cache_data
def load_background_X(csv_path='output.csv', features=DEFAULT_FEATURES, sample_n=100):
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        X = df[features].dropna(axis=0).astype(np.float32)
        if len(X) == 0:
            return None
        if len(X) > sample_n:
            X = X.sample(n=sample_n, random_state=0)
        return X
    except Exception:
        return None

def predict_survival_probs(model, X_row_df, years=PRED_YEARS):
    # 优先使用 predict_survival_function；若不可用则退化为累计风险计算
    try:
        sfs = model.predict_survival_function(X_row_df)
        sf = sfs[0]
        probs = {}
        for y in years:
            t = float(YEAR_TO_MONTHS[y])
            try:
                p = float(sf(t))
            except Exception:
                # 某些版本需要数组调用
                p = float(sf([t])[0])
            probs[y] = p
        return probs
    except Exception:
        # 退化：使用累计风险函数（S=exp(-H))
        chs = model.predict_cumulative_hazard_function(X_row_df)
        ch = chs[0]
        probs = {}
        for y in years:
            t = float(YEAR_TO_MONTHS[y])
            try:
                H = float(ch(t))
            except Exception:
                H = float(ch([t])[0])
            probs[y] = float(np.exp(-H))
        return probs


def main():
    st.sidebar.title("GBM Survival Risk Prediction (Cox GBM)")
    st.sidebar.markdown(
        "- Uses 10 features for CoxGBM survival analysis\n"
        "- Predicts 1/3/5-year survival probabilities (time unit assumed: months 12/36/60)\n"
        "- SHAP force plot explains contributions to the risk score"
    )

    # 加载模型与特征
    try:
        payload = load_model_payload()
        model = payload.get('model', None)
        features = payload.get('features', DEFAULT_FEATURES)
        if model is None:
            raise RuntimeError('Model not found: payload["model"] is None')
        st.sidebar.success("Model loaded")
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {e}")
        return

    # 校验特征与映射
    missing_maps = [f for f in features if f not in OPTIONS]
    if missing_maps:
        st.sidebar.warning(f"Missing value mapping for features: {missing_maps}")

    # 背景数据用于 SHAP（可选）
    X_bg = load_background_X('output.csv', features=features, sample_n=80)
    if X_bg is None:
        st.sidebar.info("output.csv not found or unreadable; SHAP will be slower or unavailable")
    else:
        st.sidebar.info(f"SHAP background samples: {len(X_bg)}")

    # 页面标题
    st.title("GBM Survival Prediction")
    st.markdown("Please fill the inputs below and click 'Predict'.")

    # 三列布局：按训练顺序渲染输入控件
    col1, col2, col3 = st.columns(3)
    inputs = {}
    for i, feat in enumerate(features):
        label_cn = FEATURE_LABELS_CN.get(feat, feat)
        opts = OPTIONS.get(feat, [])
        with (col1 if i % 3 == 0 else col2 if i % 3 == 1 else col3):
            if opts:
                inputs[feat] = _select_option(label_cn, opts, key=f"opt_{feat}")
            else:
                # 兜底的数字输入
                inputs[feat] = st.number_input(label_cn, value=0, step=1, key=f"num_{feat}")

    if st.button("Predict"):
        # 组装输入行（必须严格按训练顺序）
        row = [inputs[f] for f in features]
        input_df = pd.DataFrame([row], columns=features).astype(np.float32)

        # 1/3/5 年生存概率
        try:
            probs = predict_survival_probs(model, input_df, years=PRED_YEARS)
        except Exception as e:
            st.error(f"Failed to compute survival probabilities: {e}")
            return

        st.subheader("Survival probabilities (predicted)")
        cols = st.columns(len(PRED_YEARS))
        for idx, y in enumerate(PRED_YEARS):
            p = probs.get(y, np.nan)
            with cols[idx]:
                st.metric(label=f"{y}-year survival", value=f"{p*100:.2f}%" if np.isfinite(p) else "N/A")

        # 风险分数（越大风险越高）
        try:
            risk = float(model.predict(input_df)[0])
            st.markdown(f"**Risk score (relative risk)**: {risk:.4f}")
        except Exception:
            pass

        # SHAP 力图（基于风险分数）
        st.write("---")
        st.subheader("Explainability (SHAP force plot for risk score)")
        try:
            if X_bg is None or len(X_bg) == 0:
                # 退化：使用少量重复背景，计算会较慢
                X_bg_use = np.repeat(input_df.values, repeats=20, axis=0)
            else:
                X_bg_use = X_bg.values

            f = lambda X: model.predict(pd.DataFrame(X, columns=features))
            explainer = shap.KernelExplainer(f, X_bg_use)
            sv = explainer.shap_values(input_df.values, nsamples="auto")

            # 兼容不同形状
            shap_value = sv[0] if isinstance(sv, list) else (sv[0] if sv.ndim == 2 else sv.squeeze())
            expected_value = explainer.expected_value
            if isinstance(expected_value, (list, np.ndarray)):
                expected_value = float(np.mean(expected_value))

            try:
                force_plot = shap.force_plot(
                    expected_value,
                    shap_value,
                    input_df.iloc[0],
                    feature_names=[FEATURE_LABELS_CN.get(f, f) for f in features],
                    matplotlib=True,
                    show=False,
                    figsize=(20, 3)
                )
                st.pyplot(force_plot)
            except Exception as e:
                st.error(f"Failed to render SHAP force plot: {e}")
        except Exception as e:
            st.warning(f"Could not generate SHAP explanation: {e}")


if __name__ == "__main__":
    main()
