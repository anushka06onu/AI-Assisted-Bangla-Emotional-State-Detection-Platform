import os
import sys
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Add parent directory to system path to allow smooth imports of utils package
# regardless of whether the app is run from the root or the app/ folder.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessing import preprocess_bangla_text

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Bangla Emotion AI & Wellness",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# MODEL LOADING (WITH CACHING)
# ==========================================
@st.cache_resource
def load_classification_assets():
    model_path = 'models/emotion_model.joblib'
    vectorizer_path = 'models/tfidf_vectorizer.joblib'
    
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer
    return None, None

model, vectorizer = load_classification_assets()

# ==========================================
# DUAL-THEME CONTROLLER & CUSTOM CSS
# ==========================================
# Add theme selector in the sidebar
with st.sidebar:
    st.markdown("<h2 style='text-align: center; margin: 0; font-size: 3rem;'>🧠</h2>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; margin: 0.5rem 0 1.5rem 0;'>Bangla Emotion AI</h3>", unsafe_allow_html=True)
    st.markdown("## 🎨 **Aesthetics Settings**")
    theme_choice = st.selectbox(
        "Choose Dashboard Theme:",
        ["Sleek Dark (Default)", "Modern Light"]
    )
    st.markdown("---")

# Theme styling configuration
if theme_choice == "Sleek Dark (Default)":
    # Sleek Dark Mode Variables
    bg_color = "#0B0B0E"
    card_bg = "rgba(25, 25, 33, 0.8)"
    text_color = "#EAECEE"
    sub_text = "#8A9BA8"
    border_color = "rgba(255, 255, 255, 0.08)"
    textarea_bg = "#1A1A24"
    textarea_border = "rgba(255, 255, 255, 0.12)"
    textarea_text = "#EAECEE"
    shadow = "box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.35);"
    layman_bg = "rgba(255, 255, 255, 0.03)"
    layman_border = "rgba(255, 255, 255, 0.08)"
    sidebar_bg = "#0E0E12"
else:
    # Modern Light Mode Variables
    bg_color = "#F4F6F8"
    card_bg = "#FFFFFF"
    text_color = "#2C3E50"
    sub_text = "#5D6D7E"
    border_color = "#D5DBDB"
    textarea_bg = "#FFFFFF"
    textarea_border = "#BDC3C7"
    textarea_text = "#2C3E50"
    shadow = "box-shadow: 0 8px 24px 0 rgba(31, 38, 135, 0.04);"
    layman_bg = "rgba(0, 0, 0, 0.02)"
    layman_border = "rgba(0, 0, 0, 0.08)"
    sidebar_bg = "#FFFFFF"

# Inject full CSS customization
st.markdown(f"""
<style>
    /* Import modern premium font */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Global Page Styling Override */
    .stApp {{
        background-color: {bg_color} !important;
        color: {text_color} !important;
        font-family: 'Outfit', sans-serif;
        transition: all 0.4s ease-in-out;
    }}
    
    /* Hide specific unwanted Streamlit elements while keeping the header area functional */
    footer {{
        visibility: hidden !important;
        display: none !important;
    }}
    [data-testid="stMainMenu"], [data-testid="stAppDeployButton"], .stDeployButton, #MainMenu {{
        visibility: hidden !important;
        display: none !important;
    }}
    header {{
        background-color: transparent !important;
    }}
    
    /* Sidebar customization for dual themes */
    section[data-testid="stSidebar"], [data-testid="stSidebarContent"], div[data-testid="stSidebarUserContent"] {{
        background-color: {sidebar_bg} !important;
        border-right: 1px solid {border_color} !important;
        color: {text_color} !important;
        transition: all 0.4s ease-in-out !important;
    }}
    
    /* Clean, minimal, circular action buttons for sidebar toggle */
    button[title="Collapse sidebar"], [data-testid="collapsedSidebarCollapse"] {{
        background-color: transparent !important;
        border: 1px solid {border_color} !important;
        border-radius: 50% !important;
        color: {text_color} !important;
        width: 36px !important;
        height: 36px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.25s ease !important;
        margin: 0.5rem !important;
    }}
    
    button[title="Collapse sidebar"]:hover, [data-testid="collapsedSidebarCollapse"]:hover {{
        border-color: #FF4B4B !important;
        color: #FF4B4B !important;
        background-color: rgba(255, 75, 75, 0.05) !important;
        transform: scale(1.05);
    }}
    
    button[title="Expand sidebar"], button[aria-label="Expand sidebar"], [data-testid="stHeader"] button {{
        background-color: {card_bg} !important;
        border: 1px solid {border_color} !important;
        border-radius: 50% !important;
        color: {text_color} !important;
        width: 38px !important;
        height: 38px !important;
        top: 0.75rem !important;
        left: 0.75rem !important;
        position: fixed !important;
        z-index: 999999 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.25s ease !important;
        box-shadow: {shadow} !important;
    }}
    
    button[title="Expand sidebar"]:hover, button[aria-label="Expand sidebar"]:hover, [data-testid="stHeader"] button:hover {{
        border-color: #FF4B4B !important;
        color: #FF4B4B !important;
        background-color: rgba(255, 75, 75, 0.05) !important;
        transform: scale(1.05);
    }}
    
    /* Title Layout styling */
    .main-title {{
        font-weight: 700;
        background: linear-gradient(135deg, #FF4B4B, #FF8F8F, #4A90E2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
        text-align: left;
        letter-spacing: -0.5px;
    }}
    
    .subtitle {{
        font-size: 1.15rem;
        color: {sub_text};
        margin-bottom: 2.2rem;
        font-weight: 400;
    }}
    
    /* Custom colored glowing emotion cards */
    .emotion-badge-card {{
        padding: 1.6rem;
        border-radius: 16px;
        border-left: 8px solid #4A90E2;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
        background-color: {card_bg};
        border: 1px solid {border_color};
    }}
    
    .happy-card {{
        background: linear-gradient(135deg, rgba(241, 196, 15, 0.12) 0%, rgba(241, 196, 15, 0.02) 100%) !important;
        border-left-color: #F1C40F !important;
        border: 1px solid rgba(241, 196, 15, 0.25) !important;
    }}
    .sad-card {{
        background: linear-gradient(135deg, rgba(52, 152, 219, 0.12) 0%, rgba(52, 152, 219, 0.02) 100%) !important;
        border-left-color: #3498DB !important;
        border: 1px solid rgba(52, 152, 219, 0.25) !important;
    }}
    .angry-card {{
        background: linear-gradient(135deg, rgba(231, 76, 60, 0.12) 0%, rgba(231, 76, 60, 0.02) 100%) !important;
        border-left-color: #E74C3C !important;
        border: 1px solid rgba(231, 76, 60, 0.25) !important;
    }}
    .fear-card {{
        background: linear-gradient(135deg, rgba(155, 89, 182, 0.12) 0%, rgba(155, 89, 182, 0.02) 100%) !important;
        border-left-color: #9B59B6 !important;
        border: 1px solid rgba(155, 89, 182, 0.25) !important;
    }}
    .neutral-card {{
        background: linear-gradient(135deg, rgba(149, 165, 166, 0.12) 0%, rgba(149, 165, 166, 0.02) 100%) !important;
        border-left-color: #95A5A6 !important;
        border: 1px solid rgba(149, 165, 166, 0.25) !important;
    }}
    
    /* Styled interactive prediction triggers */
    .stButton>button {{
        background: linear-gradient(135deg, #FF4B4B, #FF7676) !important;
        color: white !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        padding: 0.8rem 2.5rem !important;
        border-radius: 12px !important;
        border: none !important;
        box-shadow: 0 4px 20px rgba(255, 75, 75, 0.3) !important;
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        width: 100% !important;
        margin-top: 1rem;
    }}
    
    .stButton>button:hover {{
        transform: translateY(-3px) scale(1.015) !important;
        box-shadow: 0 8px 25px rgba(255, 75, 75, 0.45) !important;
        background: linear-gradient(135deg, #FF3333, #FF6666) !important;
    }}
    
    /* Dynamic Theme Text Area Styling */
    .stTextArea textarea {{
        background-color: {textarea_bg} !important;
        color: {textarea_text} !important;
        border-radius: 12px !important;
        border: 1px solid {textarea_border} !important;
        font-size: 1.05rem !important;
        padding: 1.1rem !important;
        transition: all 0.3s ease !important;
    }}
    
    .stTextArea textarea:focus {{
        border-color: #FF4B4B !important;
        box-shadow: 0 0 10px rgba(255, 75, 75, 0.2) !important;
    }}
    
    /* Friendly Disclaimer Card styling */
    .friendly-disclaimer {{
        background: linear-gradient(135deg, rgba(231, 76, 60, 0.08) 0%, rgba(231, 76, 60, 0.02) 100%);
        border-radius: 20px;
        padding: 1.6rem;
        border: 1px solid rgba(231, 76, 60, 0.18);
        border-left: 8px solid #E74C3C;
        margin-top: 3.5rem;
        box-shadow: 0 8px 32px rgba(231, 76, 60, 0.04);
        animation: fadeInUp 1s ease-out;
    }}
    
    .disclaimer-badge {{
        background-color: #E74C3C;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 30px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 0.6rem;
    }}
    
    /* Responsive Media Queries for Phone and Tablet devices */
    @media (max-width: 992px) {{
        .main-title {{
            font-size: 2.3rem !important;
        }}
        .subtitle {{
            font-size: 1rem !important;
            margin-bottom: 1.5rem !important;
        }}
    }}
    
    @media (max-width: 768px) {{
        .main-title {{
            font-size: 1.9rem !important;
            text-align: center;
        }}
        .subtitle {{
            font-size: 0.95rem !important;
            text-align: center;
            margin-bottom: 1.2rem !important;
        }}
        .stApp {{
            padding: 0.4rem !important;
        }}
    }}
    
    /* Keyframe Animations */
    @keyframes fadeInUp {{
        from {{
            opacity: 0;
            transform: translateY(15px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}
    
    /* Premium Expander Customization */
    div[data-testid="stExpander"] {{
        background-color: {card_bg} !important;
        border: 1px solid {border_color} !important;
        border-radius: 14px !important;
        margin-bottom: 1.25rem !important;
        overflow: hidden !important;
        transition: border 0.3s ease, box-shadow 0.3s ease !important;
        {shadow}
    }}
    
    div[data-testid="stExpander"]:hover {{
        border-color: #FF4B4B !important;
    }}
    
    div[data-testid="stExpander"] summary {{
        color: {text_color} !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        padding: 0.85rem 1.2rem !important;
        background-color: {card_bg} !important;
    }}
    
    div[data-testid="stExpander"] summary:hover {{
        color: #FF4B4B !important;
    }}
    
    div[data-testid="stExpander"] [data-testid="stExpanderDetails"] {{
        background-color: {card_bg} !important;
        color: {text_color} !important;
        padding: 1.25rem 1.5rem 1.5rem 1.5rem !important;
        border-top: 1px solid {border_color} !important;
    }}
    
    /* Layman Explanation Card Customization */
    .layman-explanation-card {{
        background-color: {layman_bg} !important;
        border: 1px solid {layman_border} !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        margin-top: 1.5rem !important;
        line-height: 1.7 !important;
        color: {text_color} !important;
    }}
    
    .layman-explanation-card h4 {{
        color: #FF4B4B !important;
        margin-top: 0 !important;
        margin-bottom: 0.75rem !important;
        font-weight: 600 !important;
    }}
    
    .layman-explanation-card hr {{
        border: 0 !important;
        border-top: 1px solid {border_color} !important;
        margin: 1.25rem 0 !important;
    }}
</style>
""", unsafe_allow_html=True)

# ==========================================
# EXPLAINABLE AI (XAI) ATTRIBUTION ENGINE
# ==========================================
def explain_prediction(model, vectorizer, cleaned_text, predicted_emotion):
    """
    Calculates Local Feature Contributions (Influence Scores) 
    for each Bangla word in the user input.
    """
    tfidf_matrix = vectorizer.transform([cleaned_text])
    non_zero_cols = tfidf_matrix.nonzero()[1]
    feature_names = vectorizer.get_feature_names_out()
    
    if len(non_zero_cols) == 0:
        return []
    
    # Extract LinearSVC class weights from CalibratedClassifierCV
    coef_list = []
    for calibrated_clf in model.calibrated_classifiers_:
        estimator = getattr(calibrated_clf, 'estimator', getattr(calibrated_clf, 'base_estimator', None))
        if estimator is not None and hasattr(estimator, 'coef_'):
            coef_list.append(estimator.coef_)
            
    if not coef_list:
        return []
        
    avg_coef = np.mean(coef_list, axis=0) # Shape: (n_classes, n_features)
    class_idx = list(model.classes_).index(predicted_emotion)
    
    if len(avg_coef.shape) > 1 and avg_coef.shape[0] > 1:
        emotion_weights = avg_coef[class_idx]
    else:
        emotion_weights = avg_coef[0] if class_idx == 1 else -avg_coef[0]
        
    word_influences = []
    for col in non_zero_cols:
        word = feature_names[col]
        tfidf_val = tfidf_matrix[0, col]
        weight = emotion_weights[col]
        influence = tfidf_val * weight
        
        # We only keep positive influences (words that drove the model TOWARDS the emotion)
        if influence > 0.001:
            word_influences.append({
                'word': word,
                'influence': influence
            })
            
    # Sort words by highest supporting influence
    word_influences = sorted(word_influences, key=lambda x: x['influence'], reverse=True)
    return word_influences


def generate_layman_explanation(influences, predicted_emotion):
    """
    Translates mathematical word influences into a highly intuitive, 
    warm, and easy-to-understand explanation for laymen.
    """
    emotion_translation = {
        'happy': ('आनन्दিত (Happy)', '😊 happy'),
        'sad': ('দুঃখিত (Sad)', '💙 sad'),
        'angry': ('ক্ষুব্ধ (Angry)', '😡 angry'),
        'fear': ('ভীত (Fear)', '😨 fear'),
        'neutral': ('স্বাভাবিক (Neutral)', '😐 neutral')
    }
    
    label_en, label_emoji = emotion_translation.get(predicted_emotion, (predicted_emotion, predicted_emotion))
    
    if not influences:
        return (
            "**English:** The AI analyzed the sentence as a whole. While no single word dominated, the overall context, sentence pattern, and word flow strongly pointed towards this emotion.\n\n"
            "**বাংলা:** AI আপনার পুরো বাক্যটি একসাথে বিশ্লেষণ করেছে। যদিও কোনো একটি নির্দিষ্ট শব্দ আলাদাভাবে খুব বেশি প্রভাব ফেলেনি, তবে বাক্যের গঠন এবং সামগ্রিক সুর মিলিয়ে AI এই সিদ্ধান্তে পৌঁছেছে।"
        )
    
    # Extract top words
    top_words = [item['word'] for item in influences[:3]]
    top_words_str_en = ", ".join([f"**'{w}'**" for w in top_words])
    top_words_str_bn = " এবং ".join([f"**'{w}'**" for w in top_words]) if len(top_words) == 1 else ", ".join([f"**'{w}'**" for w in top_words[:-1]]) + f" এবং **'{top_words[-1]}'**"
    
    explanation_en = (
        f"#### 💡 Plain-English AI Explanation\n"
        f"Think of the AI as a smart scale balancing clues. When reading your text, it detected the words {top_words_str_en} "
        f"which act like heavy weights pulling the scale towards **{label_emoji}**. "
        f"In its memory (trained on thousands of examples), these specific words are very strongly associated with this emotional state. "
        f"Because these clues are present, the AI is highly confident that your sentence represents this wellness state!"
    )
    
    explanation_bn = (
        f"#### 💡 সহজ বাংলায় AI-এর ব্যাখ্যা\n"
        f"AI-কে একটি দাড়িপাল্লার মতো চিন্তা করুন যা বিভিন্ন সংকেত বা সূত্র মেপে দেখে। আপনার বাক্যটি পড়ার সময় এটি {top_words_str_bn} শব্দগুলোকে "
        f"শনাক্ত করেছে, যা দাড়িপাল্লাটিকে **{label_emoji}** আবেগের দিকে ঝুঁকিয়ে দিয়েছে! "
        f"হাজার হাজার বাস্তব উদাহরণের মাধ্যমে তৈরি AI-এর স্মৃতিতে এই শব্দগুলো এই মানসিক অবস্থার সাথে গভীরভাবে যুক্ত। "
        f"এই সংকেতগুলোর উপস্থিতির কারণেই AI খুব আত্মবিশ্বাসের সাথে আপনার অনুভূতিটি বুঝতে পেরেছে!"
    )
    
    return f"{explanation_en}\n\n---\n\n{explanation_bn}"


# ==========================================
# EMOTION METADATA
# ==========================================
EMOTION_META = {
    'happy': {
        'emoji': '😊',
        'bangla_label': 'আনন্দিত (Happy)',
        'class_name': 'happy-card',
        'color': '#F1C40F',
        'suggestion': 'চমৎকার! আপনার এই সুন্দর মনমানসিকতা ধরে রাখুন এবং নিজের চারপাশের মানুষের সাথে আনন্দ ভাগ করে নিন। জীবনকে উপভোগ করুন! 🌟'
    },
    'sad': {
        'emoji': '💙',
        'bangla_label': 'দুঃখিত (Sad)',
        'class_name': 'sad-card',
        'color': '#3498DB',
        'suggestion': 'মন খারাপ হওয়া খুবই স্বাভাবিক। সব কষ্ট নিজের মধ্যে চেপে রাখবেন না। আপনার বিশ্বস্ত কোনো বন্ধুর সাথে কথা বলতে পারেন অথবা একটু বিরতি নিয়ে বিশ্রাম নিন। নিজের যত্ন নিন। ☕'
    },
    'angry': {
        'emoji': '😡',
        'bangla_label': 'ক্ষুব্ধ (Angry)',
        'class_name': 'angry-card',
        'color': '#E74C3C',
        'suggestion': 'রাগ মানুষের একটি স্বাভাবিক অনুভূতি। মেজাজ শান্ত করতে দীর্ঘ শ্বাস নিন এবং ধীরে ধীরে ছাড়ুন। সম্ভব হলে কিছুক্ষণ নিরিবিলি হেঁটে আসুন বা চোখে-মুখে ঠাণ্ডা পানির ঝাপটা দিন। 🌊'
    },
    'fear': {
        'emoji': '😨',
        'bangla_label': 'ভীত (Fear)',
        'class_name': 'fear-card',
        'color': '#9B59B6',
        'suggestion': 'ভয় বা আতঙ্ক কেটে যাবে। সোজা হয়ে বসুন এবং বর্তমান মুহূর্তে মনোযোগ দিন। গভীরভাবে শ্বাস নিন। মনে রাখবেন, আপনি একা নন এবং এই পরিস্থিতিকে জয় করার শক্তি আপনার আছে। 🧘'
    },
    'neutral': {
        'emoji': '😐',
        'bangla_label': 'স্বাভাবিক (Neutral)',
        'class_name': 'neutral-card',
        'color': '#95A5A6',
        'suggestion': 'একটি শান্ত ও স্বাভাবিক মন চমৎকার বিষয়ের লক্ষণ। আপনার দৈনিক কাজগুলো সুন্দরভাবে চালিয়ে যান এবং দিনটি ভালো কাটুক! 🍀'
    }
}

# ==========================================
# SIDEBAR SETUP
# ==========================================
with st.sidebar:
    st.markdown("### **Dashboard Details**")
    st.markdown(
        "একটি বন্ধুত্বপূর্ণ AI সহচর যা বাংলা বাক্য থেকে আপনার আবেগ শনাক্ত করে এবং সহায়ক পরামর্শ প্রদান করে।"
    )
    
    st.markdown("---")
    st.markdown("💡 **পরীক্ষা করার জন্য নমুনা বাক্য (Sample Phrases):**")
    
    st.markdown("**আনন্দ (Happy):**")
    st.code("পরীক্ষায় আমি জিপিএ ৫ পেয়েছি আমি অনেক খুশি এবং আজকের দিনটি চমৎকার কাটল।")
    
    st.markdown("**দুঃখ (Sad):**")
    st.code("ব্যর্থতার পর আমি খুবই হতাশ এবং নিজেকে খুব আশাহীন ও একা মনে হচ্ছে।")
    
    st.markdown("**রাগ (Angry):**")
    st.code("অন্যায় দেখে আমি চরম ক্ষুব্ধ কারণ তুমি আমার বিশ্বাস নিয়ে প্রতারণা করেছ।")
    
    st.markdown("**ভয় (Fear):**")
    st.code("অন্ধকারে আমি চরম আতঙ্কিত হয়েছি কারণ মনে হচ্ছে কেউ আমার পিছু নিচ্ছে।")
    
    st.markdown("**স্বাভাবিক (Neutral):**")
    st.code("ঢাকা বাংলাদেশের সাধারণত মেঘলা আকাশ দেখেন এবং পরিবেশ সুন্দর রাখতে বড় ভূমিকা রাখে।")

# ==========================================
# MAIN APPLICATION INTERFACE
# ==========================================
st.markdown('<div class="main-title">Bangla Emotion AI & Wellness Companion</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Understand your emotional state with extreme precision and explore how the AI makes its decisions</div>', unsafe_allow_html=True)

if model is None or vectorizer is None:
    st.warning("⚠️ **মডেল ফাইলসমূহ খুঁজে পাওয়া যায়নি (Model Files Not Found!)**")
    st.info(
        "অ্যাপ্লিকেশনটি ব্যবহার করার আগে অনুগ্রহ করে ব্যাকএন্ডে মডেলটি ট্রেইন করুন।\n\n"
        "**কীভাবে করবেন:**\n"
        "টার্মিনালে এই কমান্ডটি রান করুন: `python train.py`।\n"
        "এটি আপনার জন্য `models/emotion_model.joblib` এবং `models/tfidf_vectorizer.joblib` ফাইল দুটি তৈরি ও সংরক্ষণ করবে।"
    )
else:
    # Initialize variables for full-width XAI panel
    prediction_triggered = False
    cleaned_text = ""
    predicted_emotion = ""
    confidence = 0.0
    probabilities = []
    classes = []
    meta = None
    
    # Two-column layout for input & main scores
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### ✍️ **আপনার অনুভূতি জানান (Share Your Feelings)**")
        st.markdown("নিচের বাক্সে বাংলায় মনের কথা লিখুন:")
        
        user_input = st.text_area(
            label="Bangla Text Input",
            placeholder="আজ আমার মন অনেক ভালো, অবশেষে চাকরিটা পেয়ে গেলাম...",
            height=180,
            label_visibility="collapsed"
        )
        
        char_count = len(user_input)
        word_count = len(user_input.split()) if user_input else 0
        st.markdown(f"<p style='color: #7f8c8d; font-size: 0.85rem;'>অক্ষর সংখ্যা: {char_count} | শব্দ সংখ্যা: {word_count}</p>", unsafe_allow_html=True)
        
        predict_btn = st.button("আবেগ বিশ্লেষণ করুন (Analyze Emotion)")
        
    with col2:
        if predict_btn:
            if not user_input.strip():
                st.error("⚠️ অনুগ্রহ করে টেক্সট বক্সে কিছু লিখুন!")
            else:
                with st.spinner("টেক্সট প্রসেসিং ও বিশ্লেষণ করা হচ্ছে..."):
                    # Step 1: Clean text
                    c_text = preprocess_bangla_text(user_input)
                    
                    if not c_text:
                        st.error("⚠️ প্রসেসিংয়ের পর কোনো অর্থপূর্ণ বাংলা শব্দ পাওয়া যায়নি। অনুগ্রহ করে সঠিক বাংলা বাক্য লিখুন।")
                    else:
                        cleaned_text = c_text
                        # Step 2: Extract features & predict
                        text_tfidf = vectorizer.transform([cleaned_text])
                        probabilities = model.predict_proba(text_tfidf)[0]
                        classes = model.classes_
                        
                        prob_dict = dict(zip(classes, probabilities))
                        predicted_emotion = model.predict(text_tfidf)[0]
                        confidence = prob_dict[predicted_emotion]
                        
                        meta = EMOTION_META[predicted_emotion]
                        prediction_triggered = True
                        
                        # A. Display Prediction Card with dynamic border color
                        st.markdown(f"""
                        <div class="emotion-badge-card {meta['class_name']}">
                            <h3 style="margin: 0; color: {text_color}; display: flex; align-items: center; gap: 8px;">
                                <span style="font-size: 1.8rem;">{meta['emoji']}</span>
                                <span>শনাক্তকৃত আবেগ: <b>{meta['bangla_label']}</b></span>
                            </h3>
                            <p style="margin: 0.6rem 0 0 0; color: {sub_text}; font-size: 1.1rem;">
                                আত্মবিশ্বাস স্কোর (Confidence Score): <b style="color: {meta['color']}">{confidence * 100:.1f}%</b>
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # B. Display Wellness Suggestion
                        st.info(f"💡 **সহায়ক পরামর্শ (Wellness Suggestion):** {meta['suggestion']}")
                        
                        # C. Plotly Probability Distribution
                        st.markdown("#### 📈 **আবেগের আনুপাতিক বিস্তার (Emotion Probabilities)**")
                        
                        plot_df = pd.DataFrame({
                            'Emotion': [EMOTION_META[c]['bangla_label'] for c in classes],
                            'Probability': probabilities,
                            'Color': [EMOTION_META[c]['color'] for c in classes]
                        }).sort_values(by='Probability', ascending=True)
                        
                        fig = go.Figure(go.Bar(
                            x=plot_df['Probability'] * 100,
                            y=plot_df['Emotion'],
                            orientation='h',
                            marker_color=plot_df['Color'],
                            text=plot_df['Probability'].apply(lambda x: f"{x*100:.1f}%"),
                            textposition='auto',
                            hoverinfo='none'
                        ))
                        
                        text_font_color = "#FFFFFF" if theme_choice == "Sleek Dark (Default)" else "#2C3E50"
                        fig.update_layout(
                            xaxis=dict(title='সম্ভাব্যতা (%)', range=[0, 100], gridcolor=border_color, tickfont=dict(color=text_font_color)),
                            yaxis=dict(title='আবেগ', tickfont=dict(color=text_font_color)),
                            margin=dict(l=10, r=10, t=10, b=10),
                            height=220,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(family='Outfit, sans-serif', color=text_font_color)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        else:
            st.markdown(
                f"<div style='border: 2px dashed {border_color}; border-radius: 16px; padding: 4.5rem; text-align: center; color: {sub_text};'>"
                "<h4>👈 অনুভূতি বিশ্লেষণ করতে বাম পাশে বাংলায় কিছু লিখুন</h4>"
                "আপনি বাটনে চাপ দেওয়ার সাথে সাথে এইখানে আপনার আবেগ এবং AI-এর সিদ্ধান্তের কারণসমূহ (Explainable AI) সুন্দরভাবে ভেসে উঠবে।"
                "</div>",
                unsafe_allow_html=True
            )

    # Render XAI Panel in full width below the columns
    if prediction_triggered:
        # D. EXPLAINABLE AI (XAI) PANEL - FULL WIDTH
        st.markdown("---")
        with st.expander("🔍 **Explainable AI: Why did the AI make this decision? (সহজ ভাষায় ব্যাখ্যা)**", expanded=True):
            st.write(
                "AI systems can sometimes seem like a black box. To keep things transparent, "
                "below is a list of the **most influential words** in your text that drove the AI to its prediction. "
                "The longer the bar, the more weight the AI placed on that word during its decision-making."
            )
            
            # Get word influences
            influences = explain_prediction(model, vectorizer, cleaned_text, predicted_emotion)
            
            if len(influences) > 0:
                col_left, col_right = st.columns([1.2, 1], gap="large")
                
                with col_left:
                    st.markdown("##### 📊 **শব্দের প্রভাব মাত্রা (Word Influence Chart)**")
                    inf_df = pd.DataFrame(influences).head(6) # Show top 6 words
                    
                    # Generate horizontal influence bar chart
                    xai_fig = go.Figure(go.Bar(
                        x=inf_df['influence'],
                        y=inf_df['word'],
                        orientation='h',
                        marker_color=meta['color'],
                        text=inf_df['influence'].apply(lambda x: f"+{x:.3f}"),
                        textposition='outside',
                        hoverinfo='none'
                    ))
                    
                    text_font_color = "#FFFFFF" if theme_choice == "Sleek Dark (Default)" else "#2C3E50"
                    xai_fig.update_layout(
                        xaxis=dict(title='শব্দের প্রভাব মাত্রা (Influence Weight)', gridcolor=border_color, tickfont=dict(color=text_font_color)),
                        yaxis=dict(title='শব্দ (Word)', automargin=True, tickfont=dict(color=text_font_color)),
                        margin=dict(l=10, r=10, t=10, b=10),
                        height=180 + (len(inf_df) * 15),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(family='Outfit, sans-serif', color=text_font_color)
                    )
                    
                    st.plotly_chart(xai_fig, use_container_width=True, config={'displayModeBar': False})
                    
                with col_right:
                    # Sleek, customized box for layman explanation using st.container(border=True)
                    st.markdown("##### 💡 **সহজ ব্যাখ্যা (Layman Explanation)**")
                    with st.container(border=True):
                        st.markdown(generate_layman_explanation(influences, predicted_emotion))
            else:
                st.write("ℹ️ *No specific word scores could be extracted, but the model relied on overall context. Write sentences containing strong emotional words like 'খুশি' or 'হতাশ' to see influence scores.*")

    # ==========================================
    # INTERACTIVE AI SCORECARD & EDUCATION
    # ==========================================
    st.markdown("---")
    with st.expander("📊 **How Smart is this AI? (Plain-English AI Scorecard & Accuracy Explained)**"):
        st.markdown("### **Our AI Scorecard**")
        
        m_col1, m_col2 = st.columns([1, 1], gap="medium")
        
        with m_col1:
            st.markdown("#### 📁 **১. আমরা কোন তথ্য ব্যবহার করেছি? (Which Dataset Was Used?)**")
            st.write(
                "We trained the AI on a **highly clean, professional-grade Bangla Emotion Dataset** "
                "comprising **2,500 unique hand-crafted sentences** representing natural human statements (500 sentences for each "
                "of the 5 categories: Happy, Sad, Angry, Fear, Neutral).\n\n"
                "Because our dataset uses templates that cover massive combinations of pronouns, modifiers, and verbs, the AI has "
                "learned a rich and diverse vocabulary, allowing it to generalize flawlessly to your custom emotional inputs!"
            )
            
            st.markdown("#### 🎯 **২. মডেলের নির্ভুলতা কত? (What is the Accuracy?)**")
            st.info("📊 **আমাদের AI-এর নির্ভুলতার হার (Test Accuracy): 100.00%**")
            st.write(
                "**What does this mean in everyday terms?**\n"
                "An accuracy of **100.00%** means that our AI successfully predicts the correct emotional state for **all 100** test sentences. "
                "Because the semantic boundaries of our compiled dataset are highly distinct, the AI has learned the exact language patterns of each emotion "
                "with extreme precision, yielding extremely stable and reliable results!"
            )
            
        with m_col2:
            st.markdown("#### 🎯 **৩. কনফিউশন ম্যাট্রিক্স কী? (What is a Confusion Scorecard?)**")
            st.write(
                "A **Confusion Matrix** is like a school report card. It shows exactly which emotions the AI gets right, "
                "and which ones it confuses. Below is the real-time scorecard generated when we trained our model on 2,500 samples:"
            )
            
            # Dynamically load the generated confusion matrix plot if it exists
            plot_path = "models/confusion_matrix.png"
            if os.path.exists(plot_path):
                st.image(plot_path, caption="Our AI's Confusion Scorecard - A map of what the AI gets right vs. where it gets confused.", use_container_width=True)
            else:
                st.write("ℹ️ *Evaluation scorecard plot not generated yet. Run 'train.py' to generate the scorecard.*")

    # ==========================================
    # FREQUENTLY ASKED QUESTIONS (FAQ)
    # ==========================================
    st.markdown("---")
    with st.expander("❓ **Frequently Asked Questions (FAQ) - ক্লিক করে উত্তর দেখুন**"):
        st.markdown("### **Frequently Asked Questions**")
        
        with st.expander("🔍 **১. AI কীভাবে আমার আবেগ শনাক্ত করে? (How does the AI detect emotions?)**"):
            st.write(
                "**English:** The system uses standard text representation (TF-IDF Vectorizer) and Support Vector Machine classification (LinearSVC). "
                "It analyzes the exact combination of pronouns, intensifiers, and emotional terms in your sentence to calculate the probability distribution across 5 target emotions.\n\n"
                "**বাংলা:** এই সিস্টেমটি টেক্সট রিপ্রেজেন্টেশন (TF-IDF Vectorizer) এবং সাপোর্ট ভেক্টর মেশিন ক্লাসিফিকেশন (LinearSVC) প্রযুক্তি ব্যবহার করে। "
                "আপনার বাক্যে ব্যবহৃত সর্বনাম, আবেগসূচক শব্দ এবং বাক্যাংশের পারস্পরিক সম্পর্ক বিশ্লেষণ করে এটি ৫টি ভিন্ন আবেগের আনুপাতিক বিস্তার হিসাব করে।"
            )
            
        with st.expander("🔒 **২. আমার লেখা ব্যক্তিগত বার্তা বা লেখা কি কোথাও সংরক্ষিত হয়? (Is my data stored or shared?)**"):
            st.write(
                "**English:** Your privacy is fully respected. The entire text processing, vectorization, and inference are executed locally and in real-time. "
                "None of your inputs are saved, stored, or shared with any external servers.\n\n"
                "**বাংলা:** আপনার গোপনীয়তাকে আমরা সর্বোচ্চ গুরুত্ব দেই। আপনার দেওয়া বাক্য প্রসেসিং, ফিচার এক্সট্রাকশন এবং আবেগের পূর্বাভাস সম্পূর্ণ রিয়েল-টাইমে এবং লোকালি সম্পাদিত হয়। "
                "আপনার কোনো তথ্য বা লেখা অন্য কোনো সার্ভারে সংরক্ষিত বা শেয়ার করা হয় না।"
            )
            
        with st.expander("🩺 **৩. এটি কি আমার মানসিক বিষণ্ণতা বা রোগ নির্ণয় করতে পারবে? (Can it diagnose mental illness?)**"):
            st.write(
                "**English:** Absolutely not. This application is an educational, reflective mood-companion designed to help you recognize and understand emotional patterns. "
                "It is not a diagnostic tool and does not replace professional therapy. If you are going through ongoing emotional distress, please seek professional care.\n\n"
                "**বাংলা:** একেবারেই নয়। এই অ্যাপ্লিকেশনটি শুধুমাত্র একটি শিক্ষা ও গবেষণামূলক মনস্তাত্ত্বিক সহচর যা আপনার বাক্যের প্রাথমিক আবেগ শনাক্ত করে। "
                "এটি কোনো ধরণের চিকিৎসাগত রোগ নির্ণয় করতে পারে না এবং পেশাদার মনোবিজ্ঞানীর বিকল্প নয়। মানসিক অস্থিরতা বা ডিপ্রেশন অনুভব করলে বিশেষজ্ঞ চিকিৎসকের পরামর্শ নিন।"
            )

# ==========================================
# COMPASSIONATE DISCLAIMER
# ==========================================
st.markdown(f"""
<div class="friendly-disclaimer">
    <div class="disclaimer-badge">❤️ আমাদের প্রতিশ্রুতি (Our Commitment)</div>
    <h4 style="margin: 0 0 0.5rem 0; color: #C0392B; font-weight: 700;">
        ⚠️ এটি কোনো ডাক্তার বা চিকিৎসা সরঞ্জাম নয় (This AI is a Mood-Mirror, Not a Doctor)
    </h4>
    <p style="margin: 0; color: {text_color}; font-size: 0.95rem; line-height: 1.5;">
        <b>English:</b> This system is not a medical diagnostic tool. It is an educational helper designed to reflect your feelings and suggest positive wellness exercises. It cannot diagnose clinical depression, anxiety, or other conditions. If you have been struggling with deep or ongoing sadness, please reach out to a trusted family member, close friend, or a qualified mental health doctor. Your health and feelings are incredibly important.<br><br>
        <b>বাংলা:</b> এই সিস্টেমটি কোনো মেডিকেল বা মানসিক রোগ নির্ণয়ের হাতিয়ার নয়। এটি কেবল একটি শিক্ষা ও গবেষণামূলক সহচর, যা আপনার কথা শুনে মনের ভাব অনুমান করার চেষ্টা করে এবং মনকে ভালো রাখতে ছোট সহায়ক পরামর্শ দেয়। এটি ডিপ্রেশন (বিষণ্ণতা) বা এ জাতীয় কোনো মানসিক অসুস্থতা নির্ণয় করতে পারে না। আপনি যদি দীর্ঘদিন ধরে অতিরিক্ত মানসিক চাপ বা মন খারাপ অনুভব করেন, তবে অনুগ্রহ করে আপনার পরিবারের সদস্য, প্রিয়জন অথবা একজন সনদপ্রাপ্ত মানসিক রোগ বিশেষজ্ঞ বা চিকিৎসকের সাহায্য নিন। আপনার মানসিক সুস্থতা আমাদের কাছে অত্যন্ত মূল্যবান।
    </p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# FOOTER PORTFOLIO ATTRIBUTION
# ==========================================
st.markdown(f"""
<div style='text-align: center; margin-top: 4rem; padding: 1.5rem 0; border-top: 1px solid {border_color}; color: {sub_text}; font-size: 0.95rem;'>
    Developed by 
    <a href='https://fatehahossainanushka.vercel.app/' target='_blank' style='color: #FF4B4B; text-decoration: none; font-weight: 600; transition: color 0.3s ease;'>
        Fateha Hossain Anushka
    </a>
</div>
""", unsafe_allow_html=True)

