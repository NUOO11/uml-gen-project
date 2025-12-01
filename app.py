import streamlit as st
import spacy
import nltk
from nltk.corpus import wordnet
from plantuml import PlantUML
import requests
import tarfile
import os
import shutil

# 1. 页面设置
st.set_page_config(page_title="NLP to UML Generator", layout="wide")
st.title("🎓 NLP to UML: Advanced Generation System")

# 2. 核心：自定义模型下载器 (绕过 pip 安装错误)
@st.cache_resource
def load_local_model():
    # NLTK 数据下载
    try:
        nltk.data.find('corpora/wordnet.zip')
    except LookupError:
        nltk.download('wordnet')
        nltk.download('omw-1.4')

    # 定义模型路径
    MODEL_URL = "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1.tar.gz"
    EXTRACT_PATH = "./model_data"
    MODEL_PATH = f"{EXTRACT_PATH}/en_core_web_sm-3.7.1/en_core_web_sm/en_core_web_sm-3.7.1"
    
    # 如果本地没有模型，就去下载
    if not os.path.exists(MODEL_PATH):
        st.info("正在首次下载 NLP 模型到本地目录 (约 12MB)...")
        try:
            # 下载文件
            response = requests.get(MODEL_URL, stream=True)
            if response.status_code == 200:
                with open("model.tar.gz", 'wb') as f:
                    f.write(response.raw.read())
                
                # 解压文件
                with tarfile.open("model.tar.gz", "r:gz") as tar:
                    tar.extractall(path=EXTRACT_PATH)
                
                st.success("下载并解压完成！")
            else:
                st.error("模型下载失败，请检查网络。")
                st.stop()
        except Exception as e:
            st.error(f"手动下载模型出错: {e}")
            st.stop()
    
    # 从本地路径加载模型
    try:
        nlp = spacy.load(MODEL_PATH)
        return nlp
    except Exception as e:
        st.error(f"加载模型失败: {e}")
        st.stop()

# 3. 初始化加载
with st.spinner("系统初始化中..."):
    nlp = load_local_model()

# 4. 核心逻辑 (保持不变)
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
        doc = nlp(text)
        
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN"] and token.dep_ in ["nsubj", "dobj", "pobj"]:
                if self.semantic_check_is_entity(token.lemma_):
                    class_name = token.lemma_.capitalize()
                    if class_name not in self.classes:
                        self.classes[class_name] = {'attributes': set(), 'methods': set()}
            
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

# 5. UI 界面
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
