# ────────────────────────────────────────────
# app.py – интерактивный опросник риска холецистита
# ────────────────────────────────────────────
import json, pickle, pathlib, warnings

import numpy as np
import pandas as pd
import shap
import streamlit as st
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt

# ░░░░  базовые пути  ░░░░
THIS_DIR = pathlib.Path(__file__).parent.resolve()

# ░░░░  Streamlit: конфиг и CSS  ░░░░
st.set_page_config("Прогноз холецистита", "🩺", layout="centered")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
body{
  background:linear-gradient(135deg,#6366f1 0%,#7c3aed 50%,#ec4899 100%);
  min-height:100vh;
}
/* делаем «карточку» из самого первого блока внутри страницы */
section.main > div:first-child{
  background:rgba(255,255,255,0.85);
  backdrop-filter:blur(14px);
  border-radius:1.25rem;
  box-shadow:0 10px 25px rgba(0,0,0,.15);
  padding:2.2rem 3rem;
  margin:2.5rem auto;
  max-width:720px;
}
h1.title{font-size:2rem;font-weight:700;text-align:center;margin-bottom:1.4rem;}
.subtitle{font-size:1.05rem;font-weight:600;margin:1.2rem 0 .3rem;}
.stButton>button{
  width:100%;height:3rem;border:none;border-radius:.65rem;
  font-weight:600;color:#fff;
  background:linear-gradient(90deg,#6366f1 0%,#7c3aed 100%);
  transition:background .2s;
}
.stButton>button:hover{background:#6366f1;}
</style>""",
    unsafe_allow_html=True,
)

# ░░░░  загрузка модели и прочих артефактов  ░░░░
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = CatBoostClassifier()
    model.load_model(THIS_DIR / "catboost_gb17.cbm")

    enc_map = json.loads((THIS_DIR / "enc_map.json").read_text("utf-8"))
    medians = pickle.loads((THIS_DIR / "medians.pkl").read_bytes())

    explainer = shap.TreeExplainer(model)
    return model, enc_map, medians, explainer


clf, ENC_MAP, MEDIANS, EXPL = load_artifacts()

# ░░░░  список признаков  ░░░░
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

# ░░░░  ФОРМА  ░░░░
st.markdown('<h1 class="title">🩺 Опросник риска холецистита</h1>',
            unsafe_allow_html=True)

user_vals, typed_flag = {}, {}
with st.form("input_form"):

    for feat in FEATURES:
        # подзаголовок
        st.markdown(f'<div class="subtitle">{feat}</div>', unsafe_allow_html=True)

        if feat in CATEGORICAL:  # ----- категориальные
            choice = st.selectbox(
                "выбор",                     # label обязателен, но скрываем
                options=CATEGORICAL[feat],
                key=feat,
                label_visibility="collapsed",
            )
            user_vals[feat] = choice
            typed_flag[feat] = True

        else:                     # ----- числовые
            med = float(MEDIANS.get(feat, 0.0))
            # формируем ±3σ диапазон «на глаз» (можно заменить на свои границы)
            span = 3 * (abs(med) if med else 1) + 10
            vmin, vmax = med - span, med + span
            val = st.slider(
                "число",
                min_value=float(vmin),
                max_value=float(vmax),
                value=med,
                step=0.1,
                key=feat,
                label_visibility="collapsed",
            )
            user_vals[feat] = val
            typed_flag[feat] = val != med

    submitted = st.form_submit_button("Рассчитать")

# ░░░░  ИНФЕРЕНС  ░░░░
if submitted:
    row = []
    for feat in FEATURES:
        v = user_vals[feat]
        if feat in ENC_MAP:         # строка → числовой код
            v = ENC_MAP[feat][v]
        elif not typed_flag[feat]:  # пользователь не трогал → медиана
            v = MEDIANS.get(feat, 0.0)
        row.append(v)

    df = pd.DataFrame([row], columns=FEATURES)
    prob = float(clf.predict_proba(df)[:, 1])

    st.markdown(f"### Вероятность холецистита: **{prob:.3f}**")
    if prob >= 0.5:
        st.error("💡 Модель указывает на высокий риск хронического холецистита.")
    else:
        st.success("✅ Признаков, характерных для хронического холецистита, не обнаружено.")

    # ░░░░  SHAP-бар  ░░░░
    with st.spinner("Считаем вклад признаков…"):
        shap_values = EXPL(df)
        top_idx = np.argsort(np.abs(shap_values.values[0]))[::-1][:3]

        st.markdown("#### Три наиболее влияющих признака")
        for i in top_idx:
            st.write(f"- **{FEATURES[i]}** — вклад {shap_values.values[0, i]:+0.3f}")

        # рисуем bar-plot
        plt.close("all")                              # гасим возможные старые фигуры
        fig = shap.plots.bar(shap_values, show=False) # fig – matplotlib.figure.Figure
        st.pyplot(fig)
