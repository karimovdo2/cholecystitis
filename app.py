# app.py
# Streamlit-опросник (17 признаков) ─ заглушка-прогноз холецистита
import streamlit as st
import random

st.set_page_config(page_title="Прогноз холецистита",
                   page_icon="🩺",
                   layout="centered")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    html, body, [class*="css"] {font-family:'Inter', sans-serif;}
    .stButton>button {width:100%;height:3rem;border-radius:8px;
                      background:linear-gradient(90deg,#6366f1,#7c3aed);
                      color:#fff;font-weight:600;}
    .stButton>button:hover {background:#6366f1;}
    .title {text-align:center;font-size:2rem;font-weight:700;margin-bottom:1.5rem;}
    .subtitle {font-size:1.1rem;font-weight:600;margin-top:1.4rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title">🩺 Опросник риска холецистита</div>', unsafe_allow_html=True)

# ───────────── список признаков и категории ──────────────
FEATURES = [
    'Степень фиброза по эластометрии',
    '1 блок - психическая и социальная адаптация не нарушается',
    'экстернальный тип пищевого поведения',
    'Частота приема пищи 1-2 раза -1, 3 раза -2, 4 и более раз -3',
    'St-index - индекс стеатоза ( возраст, рост, окружность талии, СД)',
    'Степень стеатоза по эластометрии',
    'Разнообразное питание да-1, нет-0',
    'Наследственность отягощена у близких родственников по ХНХ-1, ЖКБ-2, цирроз печени-3, хр.гепатит-4',
    'ОДА23+ ',
    '2 блок - интрапсихическая нарпавленностьреагирования на болезнь',
    'ИМТ',
    'Общий холестерин',
    'Перерывы между приемами пищи 2-4 часа -1, 6 и более часов-2, в разные дни значительно отличаются -3',
    'FLI - индекс стеатоза печени (рост, вес, окроужность талии, ГГТП, ТГ)',
    'ТГ',
    'Е-сигареты',
    'HSI - индекс стеатоза печени ( пол, ИМТ, АЛТ, АСТ, СД)',
]

CATEGORICAL = {
    '1 блок - психическая и социальная адаптация не нарушается': ["0 – нарушается", "1 – не нарушается"],
    'экстернальный тип пищевого поведения': ["0 – нет", "1 – есть"],
    'Частота приема пищи 1-2 раза -1, 3 раза -2, 4 и более раз -3': ["-1", "-2", "-3"],
    'Разнообразное питание да-1, нет-0': ["0 – нет", "1 – да"],
    'Наследственность отягощена у близких родственников по ХНХ-1, ЖКБ-2, цирроз печени-3, хр.гепатит-4':
        ["0 – нет", "1 – ХНХ", "2 – ЖКБ", "3 – цирроз", "4 – хр.гепатит"],
    'ОДА23+ ': ["0 – нет", "1 – да"],
    '2 блок - интрапсихическая нарпавленностьреагирования на болезнь': ["0", "1"],
    'Е-сигареты': ["0 – нет", "1 – да"],
    'Перерывы между приемами пищи 2-4 часа -1, 6 и более часов-2, в разные дни значительно отличаются -3': ["-1", "-2", "-3"],
}

# ───────────── форма ввода ──────────────
user_input = {}
with st.form("input_form", clear_on_submit=False):
    for feat in FEATURES:
        st.markdown(f'<div class="subtitle">{feat}</div>', unsafe_allow_html=True)
        if feat in CATEGORICAL:
            user_input[feat] = st.selectbox(
                " ", CATEGORICAL[feat], key=f"inp_{feat}"
            )
        else:
            user_input[feat] = st.number_input(
                " ", value=0.0, format="%.2f", key=f"inp_{feat}"
            )
    submitted = st.form_submit_button("Рассчитать")

# ───────────── заглушка-прогноз ──────────────
if submitted:
    prob = round(random.uniform(0.05, 0.95), 3)
    cls  = "✅ холецистит ОЖИДАЕТСЯ" if prob >= 0.5 else "🟢 холецистит не прогнозируется"
    st.markdown("---")
    st.markdown(f"### Вероятность холецистита: **{prob}**")
    st.success(cls)
