# ────────────────────────────────────────────
# app.py – интерактивный опросник риска холецистита
# ────────────────────────────────────────────
import pathlib, json, pickle

import numpy as np
import pandas as pd
import shap
import streamlit as st
import streamlit_shap as st_shap                # ⬅️ компонент-обёртка для SHAP
from catboost import CatBoostClassifier

THIS_DIR = pathlib.Path(__file__).parent.resolve()

# ╭────────────── UI / CSS ──────────────╮
st.set_page_config(page_title="Прогноз холецистита",
                   page_icon="🩺",
                   layout="centered")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    html, body, [class*="css"]  { font-family:'Inter', sans-serif; }
    body  { background:linear-gradient(135deg,#6366f1 0%,#7c3aed 50%,#ec4899 100%); min-height:100vh;}
    .card {max-width:720px;margin:2.5rem auto;padding:2.2rem 3rem;
           background:rgba(255,255,255,0.85);backdrop-filter:blur(16px);
           border-radius:1.25rem;box-shadow:0 10px 25px rgba(0,0,0,0.15);}
    .title {font-size:2rem;font-weight:700;text-align:center;margin-bottom:1.5rem;}
    .subtitle {font-size:1.1rem;font-weight:600;margin:1.3rem 0 0.4rem;}
    .stButton>button {width:100%;height:3rem;border-radius:0.65rem;border:none;
                      font-weight:600;color:#fff;
                      background:linear-gradient(90deg,#6366f1 0%,#7c3aed 100%);
                      transition:background 0.2s;}
    .stButton>button:hover {background:#6366f1;}
    </style>
    """,
    unsafe_allow_html=True
)

# ╭──────────── загрузка артефактов ───────────╮
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = CatBoostClassifier()
    model.load_model(THIS_DIR / "catboost_gb17.cbm")

    enc_map = json.loads((THIS_DIR / "enc_map.json").read_text(encoding="utf-8"))
    with open(THIS_DIR / "medians.pkl", "rb") as fh:
        medians = pickle.load(fh)

    explainer = shap.TreeExplainer(model)
    return model, enc_map, medians, explainer


clf, ENC_MAP, MEDIANS, EXPL = load_artifacts()

# ╭─────────── признаки ───────────╮
FEATURES = [
    'Степень фиброза по эластометрии',
    '1 блок - психическая и социальная адаптация не нарушается',
    'St-index - индекс стеатоза ( возраст, рост, окружность талии, СД)',
    'экстернальный тип пищевого поведения',
    'Частота приема пищи 1-2 раза -1, 3 раза -2, 4 и более раз -3',
    'Наследственность отягощена у близких родственников по ХНХ-1, ЖКБ-2, цирроз печени-3, хр.гепатит-4',
    'ОДА23+ ',
    '2 блок - интрапсихическая нарпавленностьреагирования на болезнь',
    'Общий холестерин',
    'Степень стеатоза по эластометрии',
    'Разнообразное питание да-1, нет-0',
    'TyG - триглицериды- глюкоза',
    'FIB-4 - индекс фиброза печени ( возраст, АЛТ, АСТ, тромбоциты)',
    'HSI - индекс стеатоза печени ( пол, ИМТ, АЛТ, АСТ, СД)',
    'ИМТ',
    'Тревога',
    'Е-сигареты'
]

CATEGORICAL = {k: list(v.keys()) for k, v in ENC_MAP.items()}

# разумные диапазоны для слайдеров
NUM_RANGES = {
    'ИМТ': (15.0, 45.0),
    'Общий холестерин': (2.5, 9.0),
    'TyG - триглицериды- глюкоза': (3.5, 6.0),
    'St-index - индекс стеатоза ( возраст, рост, окружность талии, СД)': (-2.0, 1.0),
    'HSI - индекс стеатоза печени ( пол, ИМТ, АЛТ, АСТ, СД)': (20.0, 50.0),
    'Тревога': (0, 14),
    'ОДА23+ ': (0, 100),
    'Степень фиброза по эластометрии': (0.0, 300.0),
    'Степень стеатоза по эластометрии': (0.0, 300.0),
    'FIB-4 - индекс фиброза печени ( возраст, АЛТ, АСТ, тромбоциты)': (0.0, 5.0),
}

# ╭─────────────── UI ───────────────╮
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="title">🩺 Опросник риска холецистита</div>',
                unsafe_allow_html=True)

    user_values, dirty = {}, {}

    with st.form("input_form"):
        for feat in FEATURES:
            st.markdown(f'<div class="subtitle">{feat}</div>', unsafe_allow_html=True)

            if feat in CATEGORICAL:
                sel = st.selectbox(" ", CATEGORICAL[feat], key=feat)
                user_values[feat], dirty[feat] = sel, True
            else:
                lo, hi = NUM_RANGES.get(feat, (0.0, 100.0))
                default = MEDIANS.get(feat, (lo + hi) / 2)
                val = st.slider(" ", min_value=lo, max_value=hi,
                                value=float(default), step=0.1, key=feat)
                user_values[feat], dirty[feat] = val, True
        submitted = st.form_submit_button("Рассчитать")

    if submitted:
        ordered = []
        for feat in FEATURES:
            val = user_values[feat]
            if feat in ENC_MAP:
                val = ENC_MAP[feat][val]
            ordered.append(val)

        df = pd.DataFrame([ordered], columns=FEATURES)
        prob = float(clf.predict_proba(df)[:, 1])
        label = prob >= 0.5

        st.markdown(f"### Вероятность холецистита: **{prob:.3f}**")
        (st.error if label else st.success)(
            "💡 Высокий риск хронического холецистита." if label
            else "✅ Признаков, характерных для хронического холецистита, не обнаружено."
        )

        # SHAP интерпретация
        shap_vals = EXPL(df)
        st.markdown("#### Три наиболее влияющих признака")
        top_idx = np.argsort(np.abs(shap_vals.values[0]))[::-1][:3]
        for idx in top_idx:
            st.write(f"- **{FEATURES[idx]}** — вклад {shap_vals.values[0, idx]:+0.3f}")

        with st.expander("📊 SHAP-summary-plot"):
            st_shap.st_shap(shap_vals, height=340)   # интерактивный график

    st.markdown('</div>', unsafe_allow_html=True)
