import streamlit as st
import spacy
import nltk
from nltk.corpus import wordnet
from plantuml import PlantUML
import requests
import tarfile
import os
import shutil

# ==========================================
# 1. System Config & Resources
# ==========================================
st.set_page_config(
    page_title="NLP to UML Generator",
    page_icon="🎨",
    layout="wide"
)

# Custom CSS to hide default elements and make image larger
st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; }
    img { max-width: 100%; border: 1px solid #ddd; border-radius: 5px; padding: 10px; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_resources():
    # 1. NLTK
    try:
        nltk.data.find('corpora/wordnet.zip')
    except LookupError:
        nltk.download('wordnet')
        nltk.download('omw-1.4')

    # 2. Spacy Model (Manual Download)
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
            else: st.error("Network error downloading model."); st.stop()
        except Exception as e: st.error(f"Model init failed: {e}"); st.stop()
    
    try: return spacy.load(MODEL_PATH)
    except OSError: import en_core_web_sm; return en_core_web_sm.load()

with st.spinner("⚙️ System Initializing..."):
    nlp = load_resources()

# ==========================================
# 2. Core Logic (Hybrid Extraction)
# ==========================================
class HybridUMLSystem:
    def __init__(self):
        self.classes = {}
        self.relationships = []
        self.ignored_verbs = {"be", "have", "include", "consist", "contain"}

    def check_ontology(self, word):
        # Phase 4: Semantics
        try:
            synsets = wordnet.synsets(word)
            if not synsets: return True
            return any(s.pos() == 'n' for s in synsets)
        except: return True

    def detect_multiplicity(self, token):
        # Phase 3: Multiplicity
        for child in token.children:
            if child.text.lower() in ["many", "multiple", "list", "set", "all"]: return "1..*"
            if child.tag_ == "NNS": return "0..*"
        return "1"

    def process(self, text):
        self.classes = {}
        self.relationships = []
        doc = nlp(text)
        
        # Pass 1: Entities
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN"] and token.dep_ in ["nsubj", "dobj", "pobj", "nsubjpass"]:
                if self.check_ontology(token.lemma_):
                    c_name = token.lemma_.capitalize()
                    if c_name not in self.classes:
                        self.classes[c_name] = {'attributes': set(), 'methods': set()}

        # Pass 2: Relations
        for token in doc:
            # Inheritance
            if token.lemma_ == "be":
                subj = [c for c in token.children if c.dep_ == "nsubj"]
                attr = [c for c in token.children if c.dep_ == "attr"]
                if subj and attr:
                    c = subj[0].lemma_.capitalize()
                    p = attr[0].lemma_.capitalize()
                    if c in self.classes and p in self.classes:
                        self.relationships.append((c, "<|--", p, ""))
            
            # Aggregation / Attributes
            elif token.lemma_ in ["have", "contain", "include"]:
                owners = [c for c in token.children if c.dep_ == "nsubj"]
                objs = [c for c in token.children if c.dep_ == "dobj"]
                if owners and objs:
                    owner = owners[0].lemma_.capitalize()
                    mult = self.detect_multiplicity(objs[0])
                    mult_lbl = f'"{mult}"' if mult != "1" else ""
                    if owner in self.classes:
                        obj_c = objs[0].lemma_.capitalize()
                        if obj_c in self.classes:
                            self.relationships.append((owner, "o--", obj_c, mult_lbl))
                        else:
                            self.classes[owner]['attributes'].add(objs[0].text)

            # Association / Methods
            elif token.pos_ == "VERB" and token.lemma_ not in self.ignored_verbs:
                subjs = [c for c in token.children if c.dep_ == "nsubj"]
                if subjs:
                    s_name = subjs[0].lemma_.capitalize()
                    if s_name in self.classes:
                        self.classes[s_name]['methods'].add(token.lemma_)
                        dobjs = [c for c in token.children if c.dep_ == "dobj"]
                        if dobjs:
                            o_name = dobjs[0].lemma_.capitalize()
                            if o_name in self.classes and s_name != o_name:
                                self.relationships.append((s_name, "-->", o_name, f": {token.lemma_}"))
            
            # Passive Voice
            if token.dep_ == "agent" and token.head.pos_ == "VERB":
                actual = [c for c in token.children if c.dep_ == "pobj"]
                verb = token.head
                passive = [c for c in verb.children if c.dep_ == "nsubjpass"]
                if actual and passive:
                    act = actual[0].lemma_.capitalize()
                    rec = passive[0].lemma_.capitalize()
                    if act not in self.classes: self.classes[act] = {'attributes': set(), 'methods': set()}
                    if rec not in self.classes: self.classes[rec] = {'attributes': set(), 'methods': set()}
                    self.classes[act]['methods'].add(verb.lemma_)
                    self.relationships.append((act, "-->", rec, f": {verb.lemma_}"))

        return self.generate_plantuml()

    def generate_plantuml(self):
        lines = ["@startuml", "skinparam classAttributeIconSize 0", "hide circle", "skinparam shadowing false", "skinparam ranksep 100", "skinparam nodesep 100"]
        for c, d in self.classes.items():
            lines.append(f"class {c} {{")
            for a in d['attributes']: lines.append(f"  - {a}")
            if d['attributes'] and d['methods']: lines.append("  ..")
            for m in d['methods']: lines.append(f"  + {m}()")
            lines.append("}")
        for s, r, t, l in set(self.relationships):
            lines.append(f"{s} {r} {t} {l}")
        lines.append("@enduml")
        return "\n".join(lines)

system = HybridUMLSystem()

# ==========================================
# 3. User Interface (Clean & Visual)
# ==========================================

st.title("🎓 Intelligent UML Generator")
st.markdown("Enter your requirements below, and the system will instantly generate the Class Diagram.")

# Sidebar: Evaluation (Phase 6)
with st.sidebar:
    st.header("📊 Phase 6: Evaluation")
    st.info("Validation against Ground Truth.")
    gt_input = st.text_area("Expected Classes:", value="BankSystem, Customer, Account, Administrator")
    if st.button("Run Evaluation"):
        if not system.classes: st.warning("Generate a diagram first.")
        else:
            exp = set([x.strip() for x in gt_input.split(",") if x.strip()])
            det = set(system.classes.keys())
            tp = len(exp.intersection(det))
            fp = len(det - exp); fn = len(exp - det)
            p = tp/(tp+fp) if (tp+fp) > 0 else 0
            r = tp/(tp+fn) if (tp+fn) > 0 else 0
            f1 = 2*(p*r)/(p+r) if (p+r) > 0 else 0
            st.metric("F1-Score", f"{f1:.2f}")
            st.metric("Precision", f"{p:.2f}")
            st.metric("Recall", f"{r:.2f}")

# Main Layout
input_text = st.text_area("Requirement Specification:", height=150, 
                          value="The BankSystem allows a Customer to open an Account.\nThe Account is managed by the Administrator.\nThe Customer places many Orders.")

if st.button("Generate Diagram", type="primary"):
    with st.spinner("Analyzing text and rendering diagram..."):
        # 1. Processing
        uml_code = system.process(input_text)
        
        # 2. Visualization (Center Stage)
        try:
            # 使用 HTTPS 确保图片加载
            server = PlantUML(url='https://www.plantuml.com/plantuml/img/')
            image_url = server.get_url(uml_code)
            
            st.success("✅ Diagram Generated Successfully!")
            st.image(image_url, caption="Generated UML Class Diagram", use_container_width=True)
            
            # 提供下载链接
            st.markdown(f"[📥 **Download Image**]({image_url})")
            
        except Exception as e:
            st.error("Visualization Service is busy.")
            st.markdown(f"[Click here to view image]({image_url})")
        
        # 3. Hidden Debug Info
        with st.expander("🛠️ Debug: View PlantUML Code"):
            st.code(uml_code, language='java')
