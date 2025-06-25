# ────────────────────────────────────────────
# app.py – интерактивный опросник риска холецистита
# ────────────────────────────────────────────
import json, pickle, pathlib
import numpy as np
import pandas as pd
import streamlit as st
import shap
from catboost import CatBoostClassifier

# ────────── SHAP-компонент (опц.) ──────────
try:
    import streamlit_shap as st_shap          # pip install streamlit-shap
    SHAP_AVAILABLE = True
except ModuleNotFoundError:
    SHAP_AVAILABLE = False

THIS_DIR = pathlib.Path(__file__).parent.resolve()

# ╭────────────── UI / CSS ──────────────╮
st.set_page_config(page_title="Прогноз холецистита",
                   page_icon="🩺",
                   layout="centered")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    html,body,[class*="css"]{font-family:'Inter',sans-serif;}
    body{background:linear-gradient(135deg,#6366f1 0%,#7c3aed 50%,#ec4899 100%);
         min-height:100vh;}
    .card{max-width:720px;margin:2.5rem auto;padding:2.2rem 3rem;
          background:rgba(255,255,255,0.85);backdrop-filter:blur(14px);
          border-radius:1.25rem;box-shadow:0 10px 25px rgba(0,0,0,.15);}
    .title{font-size:2rem;font-weight:700;text-align:center;margin-bottom:1.4rem;}
    .subtitle{font-size:1.05rem;font-weight:600;margin:1.2rem 0 .3rem;}
    .stButton>button{width:100%;height:3rem;border-radius:.65rem;border:none;
                     font-weight:600;color:#fff;
                     background:linear-gradient(90deg,#6366f1 0%,#7c3aed 100%);
                     transition:background .2s;}
    .stButton>button:hover{background:#6366f1;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ╭──────────── загрузка модели / метаданных ───────────╮
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = CatBoostClassifier()
    model.load_model(THIS_DIR / "catboost_gb17.cbm")

    enc_map  = json.loads((THIS_DIR / "enc_map.json").read_text("utf-8"))
    medians  = pickle.load(open(THIS_DIR / "medians.pkl", "rb"))
    expl     = shap.TreeExplainer(model)
    return model, enc_map, medians, expl


clf, ENC_MAP, MEDIANS, EXPL = load_artifacts()

# ╭────────── список признаков ──────────╮
FEATURES = [
    "Степень фиброза по эластометрии",
    "1 блок - психическая и социальная адаптация не нарушается",
    "St-index - индекс стеатоза ( возраст, рост, окружность талии, СД)",
    "экстернальный тип пищевого поведения",
    "Частота приема пищи 1-2 раза -1, 3 раза -2, 4 и более раз -3",
    "Наследственность отягощена у близких родственников по ХНХ-1, ЖКБ-2, цирроз печени-3, хр.гепатит-4",
    "ОДА23+ ",
    "2 блок - интрапсихическая нарпавленностьреагирования на болезнь",
    "Общий холестерин",
    "Степень стеатоза по эластометрии",
    "Разнообразное питание да-1, нет-0",
    "TyG - триглицериды- глюкоза",
    "FIB-4 - индекс фиброза печени ( возраст, АЛТ, АСТ, тромбоциты)",
    "HSI - индекс стеатоза печени ( пол, ИМТ, АЛТ, АСТ, СД)",
    "ИМТ",
    "Тревога",
    "Е-сигареты",
]

CATEGORICAL = {k: list(v.keys()) for k, v in ENC_MAP.items()}

# ╭────────────────────── UI ──────────────────────╮
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="title">🩺 Опросник риска холецистита</div>',
                unsafe_allow_html=True)

    user_vals, typed_flag = {}, {}

    # ───────── форма ввода ─────────
    with st.form("input_form"):
        for feat in FEATURES:
            st.markdown(f'<div class="subtitle">{feat}</div>', unsafe_allow_html=True)

            if feat in CATEGORICAL:                                # selectbox
                choice = st.selectbox(
                    label="выберите значение",
                    options=CATEGORICAL[feat],
                    key=feat,
                    label_visibility="collapsed",
                )
                user_vals[feat] = choice
                typed_flag[feat] = True

            else:                                                   # slider
                med = float(MEDIANS.get(feat, 0.0))
                span = max(1.0, 3 * abs(med) + 10)
                low, high = med - span, med + span
                val = st.slider(
                    label="укажите число",
                    min_value=low,
                    max_value=high,
                    value=med,
                    step=0.1,
                    key=feat,
                    label_visibility="collapsed",
                )
                user_vals[feat] = val
                typed_flag[feat] = not np.isclose(val, med)

        submitted = st.form_submit_button("Рассчитать")

    # ╭─────────── инференс ────────────╮
    if submitted:
        row = []
        for feat in FEATURES:
            v = user_vals[feat]
            if feat in ENC_MAP:           # code for category
                v = ENC_MAP[feat][v]
            elif not typed_flag[feat]:    # default (медиана)
                v = MEDIANS[feat]
            row.append(v)

        df = pd.DataFrame([row], columns=FEATURES)
        prob = clf.predict_proba(df)[0, 1]

        st.markdown(f"### Вероятность холецистита: **{prob:.3f}**")
        if prob >= 0.5:
            st.error("💡 Модель указывает на высокий риск хронического холецистита.")
        else:
            st.success("✅ Признаков, характерных для хронического холецистита, не обнаружено.")

        # ───────── краткий SHAP ─────────
        shap_vals = EXPL(df)
        top_idx = np.abs(shap_vals.values[0]).argsort()[::-1][:3]
        st.markdown("#### Три наиболее значимых признака")
        for i in top_idx:
            st.write(f"- **{FEATURES[i]}** — вклад {shap_vals.values[0, i]:+0.3f}")

        # полный бар-плот, если библиотека доступна
        if SHAP_AVAILABLE:
            st.markdown("#### Полная диаграмма влияния признаков")
            with st.container():
                st_shap.st_shap(shap.plots.bar(shap_vals, show=False), height=320)
        else:
            st.info("Библиотека `streamlit-shap` не установлена — интерактивный график не показан.")

    st.markdown("</div>", unsafe_allow_html=True)
