# ────────────────────────────────────────────
# app.py – интерактивный опросник риска холецистита
# ────────────────────────────────────────────
import json, pickle, pathlib, warnings
import base64, json, pickle, pathlib
import numpy as np
import pandas as pd
import shap
import streamlit as st
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt

# ░░░░  базовые пути  ░░░░
THIS_DIR = pathlib.Path(__file__).parent.resolve()
LOGO     = THIS_DIR / "hc_logo.png"
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
# ───────────── 3. Логотип первым элементом ─────────────
# ───────────── 3. Логотип первым элементом ─────────────
if LOGO.exists():
    img64 = base64.b64encode(LOGO.read_bytes()).decode()
    st.markdown(
        f"<div style='text-align:center;margin-top:1rem;margin-bottom:1rem;'>"
        f"<img src='data:image/png;base64,{img64}' width='500'>"
        f"</div>",
        unsafe_allow_html=True,
    )




# ╭──────────── загрузка артефактов ─────────────╮
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = CatBoostClassifier()
    model.load_model(THIS_DIR / "catboost_gb17.cbm")

    enc_map = json.loads((THIS_DIR / "enc_map.json").read_text(encoding="utf-8"))
    medians = pickle.loads((THIS_DIR / "medians.pkl").read_bytes())
    explainer = shap.TreeExplainer(model)
    return model, enc_map, medians, explainer

clf, ENC_MAP, MEDIANS, EXPL = load_artifacts()

# ╭──────────── признаки ─────────────╮
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


form_vals, typed = {}, {}
with st.form("hc_form"):
    for f in FEATURES:
        st.markdown(f'<div class="subtitle">{f}</div>', unsafe_allow_html=True)
        if f in CATEGORICAL:  # селектбоксы
            choice = st.selectbox(
                "выберите значение",
                CATEGORICAL[f],
                key=f,
                label_visibility="collapsed",
            )
            form_vals[f] = choice
            typed[f] = True
        else:  # числовые → слайдер
            med = float(MEDIANS.get(f, 0.0))
            span = max(abs(med), 1.0) * 3  # диапазон ±3*|median|
            val = st.slider(
                "скорректируйте значение",
                min_value=med - span,
                max_value=med + span,
                value=med,
                step=0.1,
                key=f,
                label_visibility="collapsed",
            )
            form_vals[f] = val
            typed[f] = not np.isclose(val, med)
    submitted = st.form_submit_button("Рассчитать")

if submitted:
    # формируем строку для модели
    row = []
    for f in FEATURES:
        v = form_vals[f]
        if f in ENC_MAP:  # категориальный
            v = ENC_MAP[f][v]
        elif not typed[f]:  # медиана, если не трогали
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
    top = np.argsort(np.abs(shap_values.values[0]))[::-1][:3]
    for idx in top:
        st.write(f"- **{FEATURES[idx]}** — вклад {shap_values.values[0, idx]:+0.3f}")

    # Полный бар-плот
    st.markdown("#### Вклад всех признаков")
    fig, ax = plt.subplots(figsize=(6, 4))
    shap.plots.bar(shap_values, max_display=len(FEATURES), show=False, ax=ax)
    st.pyplot(fig)

st.markdown('</div>', unsafe_allow_html=True)