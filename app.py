import streamlit as st
import spacy
import nltk
from nltk.corpus import wordnet
from plantuml import PlantUML

# 1. 页面配置
st.set_page_config(page_title="NLP to UML Generator", layout="wide")
st.title("🎓 NLP to UML: Advanced Generation System")

# 2. 资源加载 (修复版)
@st.cache_resource
def load_resources():
    # --- NLTK ---
    try:
        nltk.data.find('corpora/wordnet.zip')
    except LookupError:
        nltk.download('wordnet')
        nltk.download('omw-1.4')
    
    # --- Spacy ---
    # 这里的逻辑是：requirements.txt 已经帮我们安装了模型
    # 我们只需要告诉 Spacy 去加载这个名字即可
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        # 万一服务器没装上，这里做一个最后的兜底（虽然在云端可能会失败，但在本地有效）
        st.warning("正在尝试备用加载方案...")
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
        
    return nlp

# 显示加载状态
with st.spinner("正在初始化系统模型..."):
    try:
        nlp = load_resources()
        st.success("模型加载成功！")
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        st.stop() # 如果模型没加载成，停止运行后续代码，防止报 NameError

# 3. 核心逻辑 (AdvancedUMLSystem)
class AdvancedUMLSystem:
    def __init__(self):
        self.classes = {} 
        self.relationships = []
        self.ignored_verbs = {"be", "have", "include", "consist"}

    def semantic_check_is_entity(self, word):
        try:
            synsets = wordnet.synsets(word)
            if not synsets: return True
            return any(s.pos() == 'n' for s in synsets)
        except:
            return True

    def process(self, text):
        self.classes = {}
        self.relationships = []
        doc = nlp(text) # 使用全局加载的 nlp 对象
        
        # Rule Based Extraction
        for token in doc:
            # Classes
            if token.pos_ in ["NOUN", "PROPN"] and token.dep_ in ["nsubj", "dobj", "pobj"]:
                if self.semantic_check_is_entity(token.lemma_):
                    class_name = token.lemma_.capitalize()
                    if class_name not in self.classes:
                        self.classes[class_name] = {'attributes': set(), 'methods': set()}
            
            # Relations
            if token.lemma_ == "be":
                subjects = [c for c in token.children if c.dep_ == "nsubj"]
                attrs = [c for c in token.children if c.dep_ == "attr"]
                if subjects and attrs:
                    child = subjects[0].lemma_.capitalize()
                    parent = attrs[0].lemma_.capitalize()
                    if child in self.classes and parent in self.classes:
                        self.relationships.append((child, "<|--", parent, ""))
            
            elif token.lemma_ == "have":
                owners = [c for c in token.children if c.dep_ == "nsubj"]
                objs = [c for c in token.children if c.dep_ == "dobj"]
                if owners and objs:
                    owner_name = owners[0].lemma_.capitalize()
                    attr_name = objs[0].text
                    if owner_name in self.classes:
                        if objs[0].lemma_.capitalize() in self.classes:
                            self.relationships.append((owner_name, "o--", objs[0].lemma_.capitalize(), "has"))
                        else:
                            self.classes[owner_name]['attributes'].add(attr_name)

            elif token.pos_ == "VERB" and token.lemma_ not in self.ignored_verbs:
                subjects = [c for c in token.children if c.dep_ == "nsubj"]
                if subjects:
                    subj_name = subjects[0].lemma_.capitalize()
                    if subj_name in self.classes:
                        self.classes[subj_name]['methods'].add(token.lemma_)

        return self.generate_code()

    def generate_code(self):
        lines = ["@startuml", "skinparam classAttributeIconSize 0", "hide circle"]
        for cls_name, details in self.classes.items():
            lines.append(f"class {cls_name} {{")
            for attr in details['attributes']:
                lines.append(f"  - {attr}")
            for method in details['methods']:
                lines.append(f"  + {method}()")
            lines.append("}")
        for src, rel_type, target, label in set(self.relationships):
            label_text = f": {label}" if label else ""
            lines.append(f"{src} {rel_type} {target} {label_text}")
        lines.append("@enduml")
        return "\n".join(lines)

system = AdvancedUMLSystem()

# 4. 用户界面 UI
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Input Requirements")
    default_text = """The BankSystem allows a Customer to open an Account.
The Customer has a name.
The Administrator manages the BankSystem."""
    user_input = st.text_area("Requirements:", value=default_text, height=200)
    
    if st.button("Generate Diagram", type="primary"):
        with st.spinner("Processing..."):
            uml_code = system.process(user_input)
        
        with col1:
            with st.expander("Show PlantUML Code"):
                st.code(uml_code, language='java')
        
        with col2:
            st.subheader("📊 Diagram")
            try:
                st.image(PlantUML(url='http://www.plantuml.com/plantuml/img/').get_url(uml_code))
            except Exception as e:
                st.error(f"Image Error: {e}")
