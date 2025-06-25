# ────────────────────────────────────────────
# app.py – интерактивный опросник риска холецистита
# ────────────────────────────────────────────
import json, pickle, pathlib
import numpy as np
import pandas as pd
import streamlit as st
import shap
from catboost import CatBoostClassifier

THIS_DIR = pathlib.Path(__file__).parent.resolve()
LOGO_PATH = THIS_DIR / "hc_logo.png"     # ← логотип (512 × 512 PNG)

# ╭────────────── UI / CSS ──────────────╮
st.set_page_config("Прогноз холецистита", "🩺", layout="centered")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    html,body,[class*="css"]{font-family:'Inter',sans-serif;}
    body{background:linear-gradient(135deg,#6366f1 0%,#7c3aed 50%,#ec4899 100%);min-height:100vh;}
    .card{max-width:720px;margin:2.5rem auto;padding:2.2rem 3rem;
          background:rgba(255,255,255,0.85);backdrop-filter:blur(14px);
          border-radius:1.25rem;box-shadow:0 10px 25px rgba(0,0,0,.15);}
    .title{font-size:2rem;font-weight:700;text-align:center;margin:0.8rem 0 1.4rem;}
    .subtitle{font-size:1.05rem;font-weight:600;margin:1.2rem 0 .25rem;}
    .stButton>button{width:100%;height:3rem;border-radius:.65rem;border:none;
                     font-weight:600;color:#fff;
                     background:linear-gradient(90deg,#6366f1 0%,#7c3aed 100%);
                     transition:background .2s;}
    .stButton>button:hover{background:#6366f1;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ╭──────────── загрузка артефактов ───────────╮
@st.cache_resource(show_spinner=False)
def load_artifacts():
    # модель
    model = CatBoostClassifier()
    model.load_model(THIS_DIR / "catboost_gb17.cbm")
    # маппинг категорий
    enc_map = json.loads((THIS_DIR / "enc_map.json").read_text("utf-8"))
    # медианы числовых
    medians = pickle.loads((THIS_DIR / "medians.pkl").read_bytes())
    return model, enc_map, medians, shap.TreeExplainer(model)

clf, ENC_MAP, MEDIANS, EXPL = load_artifacts()

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

# ╭─────────────────────── UI ──────────────────────╮
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)

    # лого
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=96)

    st.markdown('<div class="title">Опросник риска холецистита</div>',
                unsafe_allow_html=True)

    user_vals, was_typed = {}, {}
    with st.form("input_form"):
        for f in FEATURES:
            st.markdown(f'<div class="subtitle">{f}</div>', unsafe_allow_html=True)
            if f in CATEGORICAL:  # категориальное
                choice = st.selectbox(" ", CATEGORICAL[f], key=f, label_visibility="collapsed")
                user_vals[f] = choice
                was_typed[f] = True
            else:                 # числовое
                med = float(MEDIANS.get(f, 0.0))
                rng = (med - 3 * abs(med) - 10, med + 3 * abs(med) + 10)
                val = st.slider(" ", min_value=rng[0], max_value=rng[1],
                                step=0.1, value=med,
                                key=f, label_visibility="collapsed")
                user_vals[f] = val
                was_typed[f] = val != med
        submitted = st.form_submit_button("Рассчитать")

    if submitted:
        # подготовка строки для модели
        row = []
        for f in FEATURES:
            v = user_vals[f]
            if f in ENC_MAP:
                v = ENC_MAP[f][v]
            elif not was_typed[f]:
                v = MEDIANS[f]
            row.append(v)
        df = pd.DataFrame([row], columns=FEATURES)

        prob = float(clf.predict_proba(df)[0, 1])
        st.markdown(f"### Вероятность холецистита: **{prob:.3f}**")
        if prob >= 0.5:
            st.error("💡 Высокий риск хронического холецистита.")
        else:
            st.success("✅ Высокого риска не обнаружено.")

        # ───────── SHAP ─────────
        shap_values = EXPL(df)
        st.markdown("#### Три наиболее влияющих признака")
        top = np.argsort(np.abs(shap_values.values[0]))[::-1][:3]
        for i in top:
            st.write(f"- **{FEATURES[i]}** — вклад {shap_values.values[0, i]:+0.3f}")

        # статичная bar-диаграмма (без streamlit_shap)
        st.markdown("#### Вклад всех признаков")
        fig, ax = plt.subplots(figsize=(6, 3))
        shap.plots.bar(shap_values, max_display=17, show=False, ax=ax)
        st.pyplot(fig)

    st.markdown("</div>", unsafe_allow_html=True)
