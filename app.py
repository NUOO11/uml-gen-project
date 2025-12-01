import streamlit as st
import spacy
import nltk
from nltk.corpus import wordnet
from plantuml import PlantUML

# 1. 页面设置
st.set_page_config(page_title="NLP to UML Generator", layout="wide")

# 2. 缓存加载模型 (服务器端优化)
@st.cache_resource
def load_models():
    try:
        nltk.data.find('corpora/wordnet.zip')
    except LookupError:
        nltk.download('wordnet')
        nltk.download('omw-1.4')
    
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    return nlp

nlp = load_models()

# 3. 核心逻辑类
class UMLGenerator:
    def __init__(self):
        self.classes = {}
        self.relationships = []

    def is_noun_entity(self, word):
        try:
            synsets = wordnet.synsets(word)
            if not synsets: return False
            return any(s.pos() == 'n' for s in synsets)
        except:
            return True

    def process(self, text):
        self.classes = {}
        self.relationships = []
        doc = nlp(text)
        
        for sent in doc.sents:
            for token in sent:
                if token.dep_ in ("nsubj", "dobj") and token.pos_ in ("NOUN", "PROPN"):
                    if self.is_noun_entity(token.lemma_):
                        cls_name = token.lemma_.capitalize()
                        if cls_name not in self.classes:
                            self.classes[cls_name] = {'methods': []}
                if token.pos_ == "VERB" and token.lemma_ not in ["be", "have"]:
                    subjects = [c for c in token.children if c.dep_ == "nsubj"]
                    if subjects:
                        cls_name = subjects[0].lemma_.capitalize()
                        if cls_name in self.classes:
                            self.classes[cls_name]['methods'].append(token.lemma_)
                            
        for sent in doc.sents:
            for token in sent:
                if token.lemma_ == "be" and token.dep_ == "ROOT":
                    subj = [c for c in token.children if c.dep_ == "nsubj"]
                    attr = [c for c in token.children if c.dep_ == "attr"]
                    if subj and attr:
                        child = subj[0].lemma_.capitalize()
                        parent = attr[0].lemma_.capitalize()
                        if child in self.classes and parent in self.classes:
                            self.relationships.append(f"{child} <|-- {parent}")

        return self.generate_code()

    def generate_code(self):
        lines = ["@startuml", "hide circle", "skinparam classAttributeIconSize 0"]
        for cls, data in self.classes.items():
            lines.append(f"class {cls} {{")
            for m in set(data['methods']):
                lines.append(f"  + {m}()")
            lines.append("}")
        lines.extend(self.relationships)
        lines.append("@enduml")
        return "\n".join(lines)

system = UMLGenerator()

# 4. 界面布局
st.title("🎓 NLP Requirements to UML Diagram")
st.markdown("Automated generation system based on **Project Phase 5** methodology.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Requirements")
    default_text = """The BankSystem allows a Customer to open an Account.
The Customer can deposit money.
A SavingsAccount is a type of Account.
The Administrator manages the System."""
    text = st.text_area("Enter text here:", value=default_text, height=200)
    btn = st.button("Generate Diagram", type="primary")

if btn and text:
    with st.spinner("Processing..."):
        uml_code = system.process(text)
        
    with col1:
        st.code(uml_code, language='java')
        
    with col2:
        st.subheader("Visualized Result")
        try:
            url = PlantUML(url='http://www.plantuml.com/plantuml/img/').get_url(uml_code)
            st.image(url)
        except Exception as e:
            st.error(f"Error rendering image: {e}")
