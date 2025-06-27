# ────────────────────────────────────────────
# app.py – интерактивный опросник риска холецистита
# ────────────────────────────────────────────
import base64, json, pickle, pathlib, warnings

import numpy as np
import pandas as pd
import shap
import streamlit as st
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt

# ░░░░ базовые пути ░░░░
THIS_DIR = pathlib.Path(__file__).parent.resolve()
LOGO     = THIS_DIR / "hc_logo.png"

# ────────── SHAP-компонент (опц.) ──────────
try:                                 # pip install streamlit-shap
    import streamlit_shap as st_shap
    SHAP_AVAILABLE = True
except ModuleNotFoundError:
    SHAP_AVAILABLE = False

# ░░░░ Streamlit: конфиг и CSS ░░░░
st.set_page_config(page_title="Прогноз холецистита",
                   page_icon=None,
                   layout="centered")


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    html,body,[class*="css"]{font-family:'Inter',sans-serif;}
    body{
      background:linear-gradient(135deg,#6366f1 0%,#7c3aed 50%,#ec4899 100%);
      min-height:100vh;
    }
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
    </style>
    """,
    unsafe_allow_html=True,
)

# ░░░░ загрузка артефактов ░░░░
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = CatBoostClassifier()
    model.load_model(THIS_DIR / "catboost_gb17.cbm")

    enc_map  = json.loads((THIS_DIR / "enc_map.json").read_text("utf-8"))
    medians  = pickle.loads((THIS_DIR / "medians.pkl").read_bytes())
    explainer = shap.TreeExplainer(model)
    return model, enc_map, medians, explainer

clf, ENC_MAP, MEDIANS, EXPL = load_artifacts()

# ░░░░ список признаков (как в модели) ░░░░
FEATURES = [
    "Степень фиброза по эластометрии",
    "1 блок - психическая и социальная адаптация не нарушается",
    "St-index - индекс стеатоза ( возраст, рост, окружность талии, СД)",
    "экстернальный тип пищевого поведения",
    "Частота приема пищи 1-2 раза -1, 3 раза -2, 4 и более раз -3",
    "Наследственность отягощена у близких родственников по ХНХ-1, ЖКБ-2, цирроз печени-3, хр.гепатит-4",
    "ОДА23+",
    "2 блок - интрапсихическая нарпавленностьреагирования на болезнь",
    "Общий холестерин",
    "Степень стеатоза по эластометрии",
    "Разнообразное питание да-1, нет-0",
    "ТГ",
    "FIB-4 - индекс фиброза печени ( возраст, АЛТ, АСТ, тромбоциты)",
    "HSI - индекс стеатоза печени ( пол, ИМТ, АЛТ, АСТ, СД)",
    "ИМТ",
    "Тревога",
    "Е-сигареты",
]
CATEGORICAL = {k: list(v.keys()) for k, v in ENC_MAP.items()}

# ░░░░ границы для числовых признаков ░░░░   ### CHG
BOUNDS = {
    "ИМТ":     (10.0, 60.0),
    "Общий холестерин": (2.0, 12.0),
    "ТГ": (0.0, 7.0),              # CHG
    "FIB-4 - индекс фиброза печени ( возраст, АЛТ, АСТ, тромбоциты)": (0.0, 2.0),   # CHG
    "HSI - индекс стеатоза печени ( пол, ИМТ, АЛТ, АСТ, СД)": (20.0, 40.0),        # CHG
    "ОДА23+": (16.0, 162.0),                                                    # NEW
    "St-index - индекс стеатоза ( возраст, рост, окружность талии, СД)": (-3.2, 51.825), # CHG
    "Степень фиброза по эластометрии": (0.0, 300.0),                             # CHG
    "Степень стеатоза по эластометрии": (0.0, 300.0),                            # CHG
}

# ░░░░ логотип ░░░░
if LOGO.exists():
    img64 = base64.b64encode(LOGO.read_bytes()).decode()
    st.markdown(
        f"<div style='text-align:center;margin-top:1rem;margin-bottom:1rem;'>"
        f"<img src='data:image/png;base64,{img64}' width='500'>"
        f"</div>",
        unsafe_allow_html=True,
    )

# ░░░░ заголовок ░░░░
st.markdown('<h1 class="title">🩺 Опросник риска холецистита</h1>', unsafe_allow_html=True)

# ░░░░ форма ░░░░
form_vals, typed = {}, {}
with st.form("hc_form"):

    # ─── ТОБОЛ – тип отношения к болезни ───  ### NEW
    st.markdown("#### ТОБОЛ — тип отношения к болезни")
    cb1 = st.checkbox("1 блок – психическая и социальная адаптация не нарушается")
    cb2 = st.checkbox("2 блок – интрапсихическая направленность реагирования на болезнь")
    cb3 = st.checkbox("3 блок – интерпсихическая направленность реагирования на болезнь (не влияет на модель)")

    form_vals["1 блок - психическая и социальная адаптация не нарушается"] = 1 if cb1 else 0
    form_vals["2 блок - интрапсихическая нарпавленностьреагирования на болезнь"] = 1 if cb2 else 0
    typed["1 блок - психическая и социальная адаптация не нарушается"] = True
    typed["2 блок - интрапсихическая нарпавленностьреагирования на болезнь"] = True
    # cb3 игнорируется в модели

    st.markdown("---")

    # ─── DEBQ – тип пищевого поведения ───  ### NEW
    st.markdown("#### DEBQ — тип пищевого поведения")
    debq_col1, debq_col2, debq_col3 = st.columns(3)
    with debq_col1:
        debq_rest = st.checkbox("Ограничительный")
    with debq_col2:
        debq_emot = st.checkbox("Эмоционогенный")
    with debq_col3:
        debq_ext  = st.checkbox("Экстернальный")

    # В модель идёт только «экстернальный»
    form_vals["экстернальный тип пищевого поведения"] = 1 if debq_ext else 0
    typed["экстернальный тип пищевого поведения"] = True

    st.markdown("---")

    # ─── Наследственность ───  ### NEW
    st.markdown("#### Наследственность (близкие родственники)")
    inh_col1, inh_col2 = st.columns(2)
    with inh_col1:
        inh_hxh   = st.checkbox("ХНХ")
        inh_jkb   = st.checkbox("ЖКБ")
    with inh_col2:
        inh_cirr  = st.checkbox("Цирроз печени")
        inh_hcv   = st.checkbox("Хронический гепатит")

    inh_code = 0
    if inh_hxh:   inh_code = 1
    if inh_jkb:   inh_code = 2
    if inh_cirr:  inh_code = 3
    if inh_hcv:   inh_code = 4
    form_vals["Наследственность отягощена у близких родственников по ХНХ-1, ЖКБ-2, цирроз печени-3, хр.гепатит-4"] = inh_code
    typed["Наследственность отягощена у близких родственников по ХНХ-1, ЖКБ-2, цирроз печени-3, хр.гепатит-4"] = True

    st.markdown("---")

    # ─── двоичные вопросы ───  ### CHG
    bin_qs = {
        "Разнообразное питание да-1, нет-0":  "Разнообразное питание?",
        "Е-сигареты":                          "Использование e‑сигарет?",
    }
    for feat, question in bin_qs.items():
        val = st.radio(question, ["Нет", "Да"], horizontal=True, index=0)
        form_vals[feat] = 1 if val == "Да" else 0
        typed[feat] = True

    st.markdown("---")

    # ─── Частота приёма пищи ───  ### NEW
    st.markdown("#### Частота приёма пищи")
    freq = st.radio(
        "Сколько раз в день вы едите?",
        ("1–2 раза", "3 раза", "4 и более раз"),
        horizontal=True,
    )
    freq_map = {"1–2 раза": -1, "3 раза": -2, "4 и более раз": -3}
    form_vals["Частота приема пищи 1-2 раза -1, 3 раза -2, 4 и более раз -3"] = freq_map[freq]
    typed["Частота приема пищи 1-2 раза -1, 3 раза -2, 4 и более раз -3"] = True

    st.markdown("---")

    # ─── Степени фиброза / стеатоза рядом ───  ### NEW
    col_f, col_s = st.columns(2)
    with col_f:
        st.markdown('<div class="subtitle">Степень фиброза по эластометрии</div>',
                    unsafe_allow_html=True)
        val_fib = st.slider(
            "fib",
            min_value=BOUNDS["Степень фиброза по эластометрии"][0],
            max_value=BOUNDS["Степень фиброза по эластометрии"][1],
            value=MEDIANS.get("Степень фиброза по эластометрии", 0.0),
            step=1.0,
            label_visibility="collapsed",
        )
        form_vals["Степень фиброза по эластометрии"] = val_fib
        typed["Степень фиброза по эластометрии"] = True

    with col_s:
        st.markdown('<div class="subtitle">Степень стеатоза по эластометрии</div>',
                    unsafe_allow_html=True)
        val_ste = st.slider(
            "ste",
            min_value=BOUNDS["Степень стеатоза по эластометрии"][0],
            max_value=BOUNDS["Степень стеатоза по эластометрии"][1],
            value=MEDIANS.get("Степень стеатоза по эластометрии", 0.0),
            step=1.0,
            label_visibility="collapsed",
        )
        form_vals["Степень стеатоза по эластометрии"] = val_ste
        typed["Степень стеатоза по эластометрии"] = True

    st.markdown("---")

    # ─── ОДА23+  (16-162)  ←--- ДОБАВЬТЕ
    st.markdown('<div class="subtitle">ОДА23+</div>', unsafe_allow_html=True)
    val_oda = st.slider(
        "oda",
        min_value=BOUNDS["ОДА23+"][0],
        max_value=BOUNDS["ОДА23+"][1],
        value=float(MEDIANS.get("ОДА23+", 16.0)),
        step=1.0,
        label_visibility="collapsed",
    )
    form_vals["ОДА23+"] = val_oda
    typed["ОДА23+"] = True

    st.markdown("---")


    # ─── остальные признаки (по списку) ───
    CUSTOM_HANDLED = {
        "1 блок - психическая и социальная адаптация не нарушается",
        "2 блок - интрапсихическая нарпавленностьреагирования на болезнь",
        "экстернальный тип пищевого поведения",
        "Наследственность отягощена у близких родственников по ХНХ-1, ЖКБ-2, цирроз печени-3, хр.гепатит-4",
        "Разнообразное питание да-1, нет-0",
        "Е-сигареты",
        "Частота приема пищи 1-2 раза -1, 3 раза -2, 4 и более раз -3",
        "Степень фиброза по эластометрии",
        "Степень стеатоза по эластометрии",
        "ОДА23+", 
    }

    for f in FEATURES:
        if f in CUSTOM_HANDLED:
            continue
        # подзаголовок
        st.markdown(f'<div class="subtitle">{f}</div>', unsafe_allow_html=True)

        # категориальные (их нет после кастомов, но на всяк. случай)
        if f in CATEGORICAL:
            choice = st.selectbox("выберите значение", CATEGORICAL[f],
                                  key=f, label_visibility="collapsed")
            form_vals[f] = choice
            typed[f] = True
        else:
            med = float(MEDIANS.get(f, 0.0))
            min_val, max_val = BOUNDS.get(f, (None, None))
            if min_val is None:
                span = max(abs(med), 1.0) * 3
                min_val = med - span
                max_val = med + span
                if med > 0:
                    min_val = max(0.0, min_val)
            val = st.slider(
                "скорректируйте значение",
                min_value=float(min_val),
                max_value=float(max_val),
                value=med,
                step=0.1,
                key=f,
                label_visibility="collapsed",
            )
            form_vals[f] = val
            typed[f] = not np.isclose(val, med)

    submitted = st.form_submit_button("Рассчитать")

# ░░░░ расчёт и визуализация ░░░░
if submitted:
    # формируем строку для модели
    row = []
    for f in FEATURES:

        v = form_vals.get(f, MEDIANS.get(f, 0.0))

        # кодируем ТОЛЬКО если это строковая метка
        if f in ENC_MAP and isinstance(v, str):
            v = ENC_MAP[f][v]

        row.append(v)


    df   = pd.DataFrame([row], columns=FEATURES)
    prob = float(clf.predict_proba(df)[:, 1])

    st.markdown(f"### Вероятность холецистита: **{prob:.3f}**")
    if prob >= 0.5:
        st.error("💡 Модель указывает на высокий риск хронического холецистита.")
    else:
        st.success("✅ Признаков, характерных для хронического холецистита, не обнаружено.")

    # ─── SHAP ───
    shap_values = EXPL(df)

    # Топ‑3 признака
    st.markdown("#### Три наиболее влияющих признака")
    top = np.argsort(np.abs(shap_values.values[0]))[::-1][:3]
    for idx in top:
        st.write(f"- **{FEATURES[idx]}** — вклад {shap_values.values[0, idx]:+0.3f}")

    # Интерактивный force‑plot
    if SHAP_AVAILABLE:
        st.markdown("#### SHAP‑график для данного пациента")
        shap_plot = shap.force_plot(
            EXPL.expected_value,
            shap_values.values[0],
            df.iloc[0],
            matplotlib=False,
        )
        st_shap.st_shap(shap_plot, height=240)
    else:
        st.info("Установите пакет `streamlit-shap`, чтобы увидеть интерактивный график.")

    # Бар‑плот с учётом знака
    st.markdown("#### Вклад всех признаков")
    vals   = shap_values.values[0]
    order  = np.argsort(np.abs(vals))[::-1]
    colors = ['#e74c3c' if v > 0 else '#1f77b4' for v in vals[order]]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(np.array(FEATURES)[order], vals[order], color=colors)
    ax.axvline(0, color="#555", linewidth=1)
    ax.invert_yaxis()
    ax.set_xlabel("SHAP value (log‑odds)")
    plt.tight_layout()
    st.pyplot(fig)

# закрывающий div
st.markdown("</div>", unsafe_allow_html=True)
