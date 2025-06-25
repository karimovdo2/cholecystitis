# ────────────────────────────────────────────
# app.py – интерактивный опросник риска холецистита
# ────────────────────────────────────────────
import json, pickle, pathlib
import numpy as np
import pandas as pd
import streamlit as st
import shap
from catboost import CatBoostClassifier

# попытка мягкого импорта streamlit-shap
try:
    import streamlit_shap as st_shap
    SHAP_AVAILABLE = True
except ModuleNotFoundError:
    SHAP_AVAILABLE = False

THIS_DIR = pathlib.Path(__file__).parent.resolve()

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

# ╭──────────── загрузка артефактов ───────────╮
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = CatBoostClassifier()
    model.load_model(THIS_DIR / "catboost_gb17.cbm")

    enc_map = json.loads((THIS_DIR / "enc_map.json").read_text(encoding="utf-8"))
    medians = pickle.loads((THIS_DIR / "medians.pkl").read_bytes())

    explainer = shap.TreeExplainer(model)
    return model, enc_map, medians, explainer


clf, ENC_MAP, MEDIANS, EXPL = load_artifacts()

# ╭─────────── признаки ───────────╮
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

    user_vals, is_typed = {}, {}
    with st.form("input"):
        for f in FEATURES:
            st.markdown(f'<div class="subtitle">{f}</div>', unsafe_allow_html=True)
            if f in CATEGORICAL:                         # категориальные
                choice = st.selectbox("", CATEGORICAL[f], key=f)
                user_vals[f] = choice
                is_typed[f] = True
            else:                                        # числовые → слайдер
                med = float(MEDIANS.get(f, 0.0))
                rng  = (med - 3*abs(med) - 10, med + 3*abs(med) + 10)
                val = st.slider("", min_value=float(rng[0]), max_value=float(rng[1]),
                                value=med, step=0.1, key=f)
                user_vals[f] = val
                is_typed[f] = val != med
        submitted = st.form_submit_button("Рассчитать")

    if submitted:
        # сбор фичей
        row = []
        for f in FEATURES:
            v = user_vals[f]
            if f in ENC_MAP:
                v = ENC_MAP[f][v]
            elif not is_typed[f]:
                v = MEDIANS[f]
            row.append(v)

        df = pd.DataFrame([row], columns=FEATURES)
        prob = float(clf.predict_proba(df)[:, 1])
        st.markdown(f"### Вероятность холецистита: **{prob:.3f}**")
        if prob >= 0.5:
            st.error("💡 Модель указывает на высокий риск хронического холецистита.")
        else:
            st.success("✅ Признаков, характерных для хронического холецистита, не обнаружено.")

        # ───── SHAP ─────
        shap_values = EXPL(df)
        st.markdown("#### Три наиболее влияющих признака")
        top_idx = np.argsort(np.abs(shap_values.values[0]))[::-1][:3]
        for i in top_idx:
            st.write(f"- **{FEATURES[i]}** — вклад {shap_values.values[0, i]:+0.3f}")

        if SHAP_AVAILABLE:
            st.markdown("#### Полная диаграмма влияния признаков")
            st_shap.st_shap(
                shap.plots.bar(shap_values, show=False),
                height=300,
            )
        else:
            st.info("`streamlit-shap` не установлен — график SHAP не показан.")

    st.markdown("</div>", unsafe_allow_html=True)
