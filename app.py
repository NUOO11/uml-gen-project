import streamlit as st
import spacy
import nltk
from nltk.corpus import wordnet
from plantuml import PlantUML
import requests
import tarfile
import os

# ==========================================
# 1. 基础配置
# ==========================================
st.set_page_config(page_title="Advanced NLP to UML System", layout="wide")

@st.cache_resource
def load_local_model():
    """手动下载并加载 Spacy 模型"""
    # NLTK
    try:
        nltk.data.find('corpora/wordnet.zip')
    except LookupError:
        nltk.download('wordnet')
        nltk.download('omw-1.4')

    # Spacy
    MODEL_URL = "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1.tar.gz"
    EXTRACT_PATH = "./model_data"
    MODEL_PATH = f"{EXTRACT_PATH}/en_core_web_sm-3.7.1/en_core_web_sm/en_core_web_sm-3.7.1"
    
    if not os.path.exists(MODEL_PATH):
        try:
            if not os.path.exists(EXTRACT_PATH):
                os.makedirs(EXTRACT_PATH)
            response = requests.get(MODEL_URL, stream=True)
            if response.status_code == 200:
                with open("model.tar.gz", 'wb') as f:
                    f.write(response.raw.read())
                with tarfile.open("model.tar.gz", "r:gz") as tar:
                    tar.extractall(path=EXTRACT_PATH)
        except Exception as e:
            st.error(f"模型加载错误: {e}")
            st.stop()
    
    try:
        return spacy.load(MODEL_PATH)
    except OSError:
        import en_core_web_sm
        return en_core_web_sm.load()

with st.spinner("🚀 System Initializing..."):
    nlp = load_local_model()

# ==========================================
# 2. 核心系统逻辑 (保持不变)
# ==========================================
class AdvancedUMLSystem:
    def __init__(self):
        self.classes = {} 
        self.relationships = []
        self.ignored_verbs = {"be", "have", "include", "consist", "contain"}
        self.ontology_suggestions = []

    def _get_lemma(self, token):
        return token.lemma_.lower()

    def semantic_check_is_entity(self, word):
        try:
            synsets = wordnet.synsets(word)
            if not synsets: return True
            return any(s.pos() == 'n' for s in synsets)
        except:
            return True

    def suggest_parent_class(self, class_name):
        try:
            syn = wordnet.synsets(class_name.lower())
            if syn:
                hypernyms = syn[0].hypernyms()
                if hypernyms:
                    parent = hypernyms[0].lemmas()[0].name().capitalize()
                    if parent not in ["Entity", "Object", "Whole", "Physical_entity"]:
                        return parent
        except:
            return None
        return None

    def check_multiplicity(self, token):
        for child in token.children:
            if child.text.lower() in ["many", "multiple", "list", "set", "collection", "all"]:
                return "1..*"
            if child.tag_ == "NNS":
                return "0..*"
        return "1"

    def process(self, text):
        self.classes = {}
        self.relationships = []
        self.ontology_suggestions = []
        doc = nlp(text)
        
        # Extraction Logic
        for token in doc:
            # Entities
            if token.pos_ in ["NOUN", "PROPN"] and token.dep_ in ["nsubj", "dobj", "pobj", "nsubjpass"]:
                if self.semantic_check_is_entity(token.lemma_):
                    class_name = token.lemma_.capitalize()
                    if class_name not in self.classes:
                        self.classes[class_name] = {'attributes': set(), 'methods': set()}
                        parent = self.suggest_parent_class(class_name)
                        if parent:
                            self.ontology_suggestions.append(f"💡 Ontology Hint: '{class_name}' is a type of '{parent}'")
            
            # Relations
            if token.lemma_ == "be":
                subjects = [c for c in token.children if c.dep_ == "nsubj"]
                attrs = [c for c in token.children if c.dep_ == "attr"]
                if subjects and attrs:
                    child = subjects[0].lemma_.capitalize()
                    parent = attrs[0].lemma_.capitalize()
                    if child in self.classes and parent in self.classes:
                        self.relationships.append((child, "<|--", parent, ""))
            
            elif token.lemma_ in ["have", "contain", "include"]:
                owners = [c for c in token.children if c.dep_ == "nsubj"]
                objs = [c for c in token.children if c.dep_ == "dobj"]
                if owners and objs:
                    owner_name = owners[0].lemma_.capitalize()
                    mult = self.check_multiplicity(objs[0])
                    mult_label = f'"{mult}"' if mult != "1" else ""
                    if owner_name in self.classes:
                        obj_lemma = objs[0].lemma_.capitalize()
                        if obj_lemma in self.classes:
                            self.relationships.append((owner_name, "o--", obj_lemma, mult_label))
                        else:
                            self.classes[owner_name]['attributes'].add(objs[0].text)

            elif token.pos_ == "VERB" and token.lemma_ not in self.ignored_verbs:
                subjects = [c for c in token.children if c.dep_ == "nsubj"]
                if subjects:
                    subj = subjects[0].lemma_.capitalize()
                    if subj in self.classes:
                        self.classes[subj]['methods'].add(token.lemma_)
                        objs = [c for c in token.children if c.dep_ == "dobj"]
                        if objs:
                            obj = objs[0].lemma_.capitalize()
                            if obj in self.classes and subj != obj:
                                self.relationships.append((subj, "-->", obj, f": {token.lemma_}"))

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

        return self.generate_code()

    def generate_code(self):
        lines = ["@startuml", "skinparam classAttributeIconSize 0", "hide circle", "skinparam shadowing false"]
        for cls_name, details in self.classes.items():
            lines.append(f"class {cls_name} {{")
            for attr in details['attributes']:
                lines.append(f"  - {attr}")
            if details['attributes'] and details['methods']: lines.append("  ..")
            for method in details['methods']:
                lines.append(f"  + {method}()")
            lines.append("}")
        for src, rel_type, target, label in set(self.relationships):
            lines.append(f"{src} {rel_type} {target} {label}")
        lines.append("@enduml")
        return "\n".join(lines)

system = AdvancedUMLSystem()

# ==========================================
# 3. UI 界面 (修复图片显示)
# ==========================================
st.title("🎓 NLP to UML Generation System")
st.markdown("**Methodology:** Hybrid Extraction (Rules + Semantics) | **Phase:** 3, 4, 5 & 6")

with st.sidebar:
    st.header("📊 Phase 6: Evaluation")
    ground_truth = st.text_area("Expected Classes:", value="BankSystem, Customer, Account")
    if st.button("Calculate Metrics"):
        if system.classes:
            expected = set([x.strip() for x in ground_truth.split(",") if x.strip()])
            detected = set(system.classes.keys())
            tp = len(expected.intersection(detected))
            fp = len(detected - expected)
            fn = len(expected - detected)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            st.metric("F1-Score", f"{f1:.2f}")

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("📝 Requirements Input")
    default_text = "The BankSystem allows a Customer to open an Account.\nThe Account is managed by the Administrator."
    user_input = st.text_area("Enter text:", value=default_text, height=250)
    if st.button("Generate Diagram", type="primary"):
        with st.spinner("Processing..."):
            uml_code = system.process(user_input)
        
        with col1:
            if system.ontology_suggestions:
                st.info(f"Ontology Suggestions: {system.ontology_suggestions}")
            with st.expander("View Code"):
                st.code(uml_code, language='java')

        with col2:
            st.subheader("📊 Class Diagram")
            
            # --- 图片显示修复逻辑 ---
            try:
                # 1. 强制使用 HTTPS
                server = PlantUML(url='https://www.plantuml.com/plantuml/img/')
                image_url = server.get_url(uml_code)
                
                # 2. 直接显示图片
                st.image(image_url, caption="Generated Diagram")
                
                # 3. [关键] 同时提供直接链接，防止图片裂开
                st.markdown(f"**[🔗 如果图片未显示，请点击这里直接查看]({image_url})**")
                
            except Exception as e:
                st.error(f"无法生成图片: {e}")
