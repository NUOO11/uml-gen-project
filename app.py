import streamlit as st
import spacy
import nltk
from nltk.corpus import wordnet
from plantuml import PlantUML
import requests
import tarfile
import os

# ==========================================
# 1. System Config
# ==========================================
st.set_page_config(page_title="NLP to UML Generator", page_icon="🎨", layout="wide")

# CSS 优化：让图片居中并自适应大小，隐藏多余的空白
st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; }
    .stImage { text-align: center; }
    img { max-width: 90%; border: 1px solid #e6e6e6; border-radius: 8px; padding: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_resources():
    try:
        nltk.data.find('corpora/wordnet.zip')
    except LookupError:
        nltk.download('wordnet')
        nltk.download('omw-1.4')

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
        except: pass # Silent fail, try fallback
    
    try: return spacy.load(MODEL_PATH)
    except: import en_core_web_sm; return en_core_web_sm.load()

nlp = load_resources()

# ==========================================
# 2. Logic (Hybrid)
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
                    mlabel = f'"{mult}"' if mult != "1" else ""
                    if o in self.classes:
                        obj_c = objs[0].lemma_.capitalize()
                        if obj_c in self.classes: self.relationships.append((o, "o--", obj_c, mlabel))
                        else: self.classes[o]['attributes'].add(objs[0].text)

            elif token.pos_ == "VERB" and token.lemma_ not in self.ignored_verbs:
                subjs = [c for c in token.children if c.dep_ == "nsubj"]
                if subjs:
                    s = subjs[0].lemma_.capitalize()
                    if s in self.classes:
                        self.classes[s]['methods'].add(token.lemma_)
                        dobjs = [c for c in token.children if c.dep_ == "dobj"]
                        if dobjs:
                            o = dobjs[0].lemma_.capitalize()
                            if o in self.classes and s != o: self.relationships.append((s, "-->", o, f": {token.lemma_}"))

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
        lines = ["@startuml", "skinparam classAttributeIconSize 0", "hide circle", "skinparam shadowing false", "skinparam ranksep 80", "skinparam nodesep 80", "skinparam linetype ortho"]
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
# 3. UI (Robust Image Rendering)
# ==========================================
st.title("🎓 Intelligent UML Generator")

# Sidebar
with st.sidebar:
    st.header("📊 Phase 6: Evaluation")
    gt = st.text_area("Expected Classes:", "BankSystem, Customer, Account, Administrator")
    if st.button("Evaluate"):
        if not system.classes: st.warning("Generate first.")
        else:
            exp = set([x.strip() for x in gt.split(",") if x.strip()])
            det = set(system.classes.keys())
            tp = len(exp.intersection(det))
            fp = len(det - exp); fn = len(exp - det)
            p = tp/(tp+fp) if (tp+fp)>0 else 0
            r = tp/(tp+fn) if (tp+fn)>0 else 0
            f1 = 2*(p*r)/(p+r) if (p+r)>0 else 0
            st.metric("F1-Score", f"{f1:.2f}")
            st.metric("Precision", f"{p:.2f}")

# Main
txt = st.text_area("Requirements:", "The BankSystem allows a Customer to open an Account.\nThe Account is managed by the Administrator.", height=150)

if st.button("Generate Diagram", type="primary"):
    with st.spinner("Processing..."):
        uml_code = system.process(txt)
        
        # --- 关键修改：直接生成 URL，使用 Markdown 渲染 (Browser-Side Rendering) ---
        try:
            # 使用 SVG 格式，清晰度更高，且使用 HTTPS
            plantuml_server = PlantUML(url='https://www.plantuml.com/plantuml/svg/')
            image_url = plantuml_server.get_url(uml_code)
            
            st.success("✅ Generation Complete!")
            
            # 使用 Markdown 语法显示图片
            # 这种方式由你的浏览器直接加载图片，绕过 Streamlit 服务器
            st.markdown(f'<div style="text-align: center;"><img src="{image_url}" alt="UML Diagram" width="100%"></div>', unsafe_allow_html=True)
            
            st.markdown(f"**[🔗 Click to Open High-Res Image]({image_url})**")
            
        except Exception as e:
            st.error(f"Error generating URL: {e}")
        
        # 调试代码（默认折叠）
        with st.expander("🛠️ View PlantUML Code"):
            st.code(uml_code, language='java')
