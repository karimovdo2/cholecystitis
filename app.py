# ────────────────────────────────────────────
# app.py – интерактивный опросник риска холецистита
# автор: <вы>
# ────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import pickle, json, pathlib
from catboost import CatBoostClassifier
import shap

THIS_DIR = pathlib.Path(__file__).parent.resolve()

# ╭─────────────────────── UI / CSS ───────────────────────╮
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
    </style>""",
    unsafe_allow_html=True
)

# ╭────────────────── загрузка артефактов ──────────────────╮
@st.cache_resource(show_spinner=False)
def load_artifacts():
    # 1. модель
    model = CatBoostClassifier()
    model.load_model(THIS_DIR / "catboost_gb17.cbm")
    # 2. категориальные признаки (список str)
    cat_feats = [
        '1 блок - психическая и социальная адаптация не нарушается',
        'экстернальный тип пищевого поведения',
        'Частота приема пищи 1-2 раза -1, 3 раза -2, 4 и более раз -3',
        'Разнообразное питание да-1, нет-0',
        'Наследственность отягощена у близких родственников по ХНХ-1, ЖКБ-2, цирроз печени-3, хр.гепатит-4',
        'ОДА23+ ',
        '2 блок - интрапсихическая нарпавленностьреагирования на болезнь',
        'Е-сигареты',
        'Перерывы между приемами пищи 2-4 часа -1, 6 и более часов-2, в разные дни значительно отличаются -3'
    ]
    # 3. сопоставление «текст в selectbox → исходный код»
    enc_map = json.loads((THIS_DIR / "enc_map.json").read_text(encoding="utf-8"))
    # 4. медианы для числовых полей
    with open(THIS_DIR / "medians.pkl", "rb") as fh:
        medians = pickle.load(fh)
    # 5. shap‑explainer (лёгкий, поэтому кэшируем)
    explainer = shap.TreeExplainer(model)
    return model, cat_feats, enc_map, medians, explainer

clf, CAT_FEATURES, ENC_MAP, MEDIANS, EXPL = load_artifacts()

# ╭─────────────────── список признаков ────────────────────╮
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

# выпадающие списки
CATEGORICAL = {k: list(v.keys()) for k, v in ENC_MAP.items()}

# ╭─────────────────────── UI ───────────────────────────────╮
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="title">🩺 Опросник риска холецистита</div>',
                unsafe_allow_html=True)

    user_values = {}
    dirty_flag = {}           # отмечаем, вводил ли пользователь значение

    with st.form("input_form"):
        for feat in FEATURES:
            st.markdown(f'<div class="subtitle">{feat}</div>',
                        unsafe_allow_html=True)
            if feat in CATEGORICAL:
                choice = st.selectbox(" ", CATEGORICAL[feat], key=feat)
                user_values[feat] = choice
                dirty_flag[feat] = True            # всегда определено
            else:
                val = st.number_input(" ", value=0.0, format="%.2f", key=feat)
                user_values[feat] = val
                dirty_flag[feat] = st.session_state[feat] != 0.0
        submitted = st.form_submit_button("Рассчитать")

    # ╭───────────── инференс ─────────────╮
    if submitted:
        # сбор в правильном порядке
        ordered = []
        for feat in FEATURES:
            val = user_values[feat]
            if feat in ENC_MAP:
                val = ENC_MAP[feat][val]           # строка → код
            else:
                if not dirty_flag[feat]:
                    val = MEDIANS.get(feat, 0.0)   # подставляем медиану
            ordered.append(val)

        df = pd.DataFrame([ordered], columns=FEATURES)
        prob = float(clf.predict_proba(df)[:, 1])
        label = prob >= 0.5

        st.markdown(f"### Вероятность холецистита: **{prob:0.3f}**")
        if label:
            st.error("💡 Модель указывает на высокий риск хронического холецистита.")
        else:
            st.success("✅ Признаков, характерных для хронического холецистита, не обнаружено.")

        # ───── краткая интерпретация SHAP ─────
        shap_vals = EXPL(df)
        st.markdown("#### Три наиболее влияющих признака")
        top = np.argsort(np.abs(shap_vals.values[0]))[::-1][:3]
        for idx in top:
            feat = FEATURES[idx]
            st.write(f"- **{feat}** — вклад {shap_vals.values[0, idx]:+0.3f}")

    st.markdown('</div>', unsafe_allow_html=True)
