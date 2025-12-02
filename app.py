import streamlit as st
import spacy
import nltk
from nltk.corpus import wordnet
from plantuml import PlantUML
import requests
import tarfile
import os

# ==========================================
# 1. 配置与资源
# ==========================================
st.set_page_config(page_title="NLP to UML Generator", page_icon="🎨", layout="wide")

st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; }
    .stImage { text-align: center; }
    img { max-width: 90%; border: 1px solid #e6e6e6; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

# Session State 初始化
if 'generated_classes' not in st.session_state:
    st.session_state['generated_classes'] = {}
if 'uml_code' not in st.session_state:
    st.session_state['uml_code'] = ""

@st.cache_resource
def load_resources():
    try:
        nltk.data.find('corpora/wordnet.zip')
    except LookupError:
        nltk.download('wordnet'); nltk.download('omw-1.4')

    MODEL_URL = "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1.tar.gz"
    EXTRACT_PATH = "./model_data"
    MODEL_PATH = f"{EXTRACT_PATH}/en_core_web_sm-3.7.1/en_core_web_sm/en_core_web_sm-3.7.1"
    
    if not os.path.exists(MODEL_PATH):
        try:
            if not os.path.exists(EXTRACT_PATH): os.makedirs(EXTRACT_PATH)
            response = requests.get(MODEL_URL, stream=True)
            if response.status_code == 200:
                with open("model.tar.gz", 'wb') as f: f.write(response.raw.read())
                with tarfile.open("model.tar.gz", "r:gz") as tar: tar.extractall(path=EXTRACT_PATH)
        except: pass
    
    try: return spacy.load(MODEL_PATH)
    except: import en_core_web_sm; return en_core_web_sm.load()

nlp = load_resources()

# ==========================================
# 2. 核心逻辑 (Hybrid)
# ==========================================
class HybridUMLSystem:
    def __init__(self):
        self.classes = {}
        self.relationships = []
        self.ignored_verbs = {"be", "have", "include", "consist", "contain", "involve"}

    def check_ontology(self, word):
        try:
            synsets = wordnet.synsets(word)
            if not synsets: return True
            return any(s.pos() == 'n' for s in synsets)
        except: return True

    def detect_multiplicity(self, token):
        for child in token.children:
            if child.text.lower() in ["many", "multiple", "list", "set", "all"]: return "1..*"
            if child.tag_ == "NNS": return "0..*"
        return "1"

    def process(self, text):
        self.classes = {}
        self.relationships = []
        doc = nlp(text)
        
        for token in doc:
            # Entities
            if token.pos_ in ["NOUN", "PROPN"] and token.dep_ in ["nsubj", "dobj", "pobj", "nsubjpass"]:
                if self.check_ontology(token.lemma_):
                    c = token.lemma_.capitalize()
                    if c not in self.classes: self.classes[c] = {'attributes': set(), 'methods': set()}

            # Relations
            if token.lemma_ == "be":
                subj = [c for c in token.children if c.dep_ == "nsubj"]
                attr = [c for c in token.children if c.dep_ == "attr"]
                if subj and attr:
                    c, p = subj[0].lemma_.capitalize(), attr[0].lemma_.capitalize()
                    if c in self.classes and p in self.classes: self.relationships.append((c, "<|--", p, ""))
            
            elif token.lemma_ in ["have", "contain", "include"]:
                owners = [c for c in token.children if c.dep_ == "nsubj"]
                objs = [c for c in token.children if c.dep_ == "dobj"]
                if owners and objs:
                    o = owners[0].lemma_.capitalize()
                    mult = self.detect_multiplicity(objs[0])
                    mlabel = f'"{mult}"' if mult != "
