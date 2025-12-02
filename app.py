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
# 1. 基础配置与模型加载器 (Stable Loader)
# ==========================================
st.set_page_config(page_title="Advanced NLP to UML System", layout="wide")

@st.cache_resource
def load_local_model():
    """
    手动下载并加载 Spacy 模型，避开云端安装权限问题。
    """
    # 1. NLTK 资源
    try:
        nltk.data.find('corpora/wordnet.zip')
    except LookupError:
        nltk.download('wordnet')
        nltk.download('omw-1.4')

    # 2. Spacy 模型手动下载
    MODEL_URL = "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1.tar.gz"
    EXTRACT_PATH = "./model_data"
    # 解压后的具体路径可能因版本不同而异，这里指向标准路径
    MODEL_PATH = f"{EXTRACT_PATH}/en_core_web_sm-3.7.1/en_core_web_sm/en_core_web_sm-3.7.1"
    
    if not os.path.exists(MODEL_PATH):
        print("正在下载模型文件...") # 终端提示
        try:
            if not os.path.exists(EXTRACT_PATH):
                os.makedirs(EXTRACT_PATH)
            
            response = requests.get(MODEL_URL, stream=True)
            if response.status_code == 200:
                with open("model.tar.gz", 'wb') as f:
                    f.write(response.raw.read())
                with tarfile.open("model.tar.gz", "r:gz") as tar:
                    tar.extractall(path=EXTRACT_PATH)
            else:
                st.error("模型下载失败 (Network Error)")
                st.stop()
        except Exception as e:
            st.error(f"模型解压失败: {e}")
            st.stop()
    
    try:
        nlp = spacy.load(MODEL_PATH)
        return nlp
    except OSError:
        # 最后的兜底：如果手动路径不对，尝试直接load包名（针对本地环境）
        import en_core_web_sm
        return en_core_web_sm.load()

# 初始化加载
with st.spinner("🚀 System Initializing (Loading NLP Models)..."):
    nlp = load_local_model()

# ==========================================
# 2. 增强版核心系统 (Enhanced Core Logic)
# ==========================================

class AdvancedUMLSystem:
    def __init__(self):
        self.classes = {} 
        self.relationships = []
        self.ignored_verbs = {"be", "have", "include", "consist", "contain"}
        self.ontology_suggestions = [] # 存储本体建议 (Phase 4)

    def _get_lemma(self, token):
        return token.lemma_.lower()

    def semantic_check_is_entity(self, word):
        """Phase 4: 简单的语义检查"""
        try:
            synsets = wordnet.synsets(word)
            if not synsets: return True
            return any(s.pos() == 'n' for s in synsets)
        except:
            return True

    def suggest_parent_class(self, class_name):
        """Phase 4 Enhancement: 使用 WordNet 建议父类"""
        try:
            syn = wordnet.synsets(class_name.lower())
            if syn:
                hypernyms = syn[0].hypernyms()
                if hypernyms:
                    parent = hypernyms[0].lemmas()[0].name().capitalize()
                    # 过滤掉太通用的词 (如 Entity, Object)
                    if parent not in ["Entity", "Object", "Whole", "Physical_entity"]:
                        return parent
        except:
            pass
        return None

    def check_multiplicity(self, token):
        """规则增强: 检测多重性 (1..*)"""
        for child in token.children:
            if child.text.lower() in ["many", "multiple", "list", "set", "collection", "all"]:
                return "1..*"
            if child.tag_ == "NNS": # 复数名词
                return "0..*"
        return "1"

    def process(self, text):
        self.classes = {}
        self.relationships = []
        self.ontology_suggestions = []
        doc = nlp(text)
        
        # --- Pass 1: 基础实体识别 ---
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN"] and token.dep_ in ["nsubj", "dobj", "pobj", "nsubjpass"]:
                if self.semantic_check_is_entity(token.lemma_):
                    class_name = token.lemma_.capitalize()
                    if class_name not in self.classes:
                        self.classes[class_name] = {'attributes': set(), 'methods': set()}
                        # 触发本体建议
                        parent = self.suggest_parent_class(class_name)
                        if parent:
                            self.ontology_suggestions.append(f"💡 Ontology Hint: '{class_name}' is a type of '{parent}'")

        # --- Pass 2: 关系与细节提取 (增强版) ---
        for token in doc:
            
            # [规则 1] 继承 (Inheritance)
            if token.lemma_ == "be":
                subjects = [c for c in token.children if c.dep_ == "nsubj"]
                attrs = [c for c in token.children if c.dep_ == "attr"]
                if subjects and attrs:
                    child = subjects[0].lemma_.capitalize()
                    parent = attrs[0].lemma_.capitalize()
                    if child in self.classes and parent in self.classes:
                        self.relationships.append((child, "<|--", parent, ""))

            # [规则 2] 聚合/属性 (Aggregation vs Attribute) + 多重性
            elif token.lemma_ in ["have", "contain", "include"]:
                owners = [c for c in token.children if c.dep_ == "nsubj"]
                objs = [c for c in token.children if c.dep_ == "dobj"]
                if owners and objs:
                    owner_name = owners[0].lemma_.capitalize()
                    
                    # 检查多重性
                    multiplicity = self.check_multiplicity(objs[0])
                    mult_label = f'"{multiplicity}"' if multiplicity != "1" else ""
                    
                    if owner_name in self.classes:
                        obj_lemma = objs[0].lemma_.capitalize()
                        # 如果宾语也是一个类，则是聚合关系
                        if obj_lemma in self.classes:
                            self.relationships.append((owner_name, "o--", obj_lemma, mult_label))
                        else:
                            # 否则是属性
                            self.classes[owner_name]['attributes'].add(objs[0].text)

            # [规则 3] 普通关联 (Association) & 方法
            elif token.pos_ == "VERB" and token.lemma_ not in self.ignored_verbs:
                subjects = [c for c in token.children if c.dep_ == "nsubj"]
                if subjects:
                    subj_name = subjects[0].lemma_.capitalize()
                    if subj_name in self.classes:
                        self.classes[subj_name]['methods'].add(token.lemma_)
                        
                        # 检查关联对象
                        objects = [c for c in token.children if c.dep_ == "dobj"]
                        if objects:
                            obj_name = objects[0].lemma_.capitalize()
                            if obj_name in self.classes and subj_name != obj_name:
                                self.relationships.append((subj_name, "-->", obj_name, f": {token.lemma_}"))

            # [规则 4 - 新增] 被动语态 (Passive Voice)
            # e.g., "The Account is managed by the Administrator."
            if token.dep_ == "agent" and token.head.pos_ == "VERB":
                # token is "by"
                actual_subj_tokens = [c for c in token.children if c.dep_ == "pobj"]
                verb = token.head
                passive_subj_tokens = [c for c in verb.children if c.dep_ == "nsubjpass"]
                
                if actual_subj_tokens and passive_subj_tokens:
                    actor = actual_subj_tokens[0].lemma_.capitalize() # Administrator
                    receiver = passive_subj_tokens[0].lemma_.capitalize() # Account
                    method = verb.lemma_ # manage
                    
                    # 确保类存在
                    for name in [actor, receiver]:
                        if name not in self.classes:
                            self.classes[name] = {'attributes': set(), 'methods': set()}
                    
                    self.classes[actor]['methods'].add(method)
                    self.relationships.append((actor, "-->", receiver, f": {method}"))

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
        
        unique_rels = set(self.relationships)
        for src, rel_type, target, label in unique_rels:
            lines.append(f"{src} {rel_type} {target} {label}")
            
        lines.append("@enduml")
        return "\n".join(lines)

system = AdvancedUMLSystem()

# ==========================================
# 3. 用户界面 (UI) - 含 Evaluation Sidebar
# ==========================================

st.title("🎓 NLP to UML Generation System")
st.markdown("**Methodology:** Hybrid Extraction (Rules + Semantics) | **Phase:** 3, 4, 5 & 6")

# --- 侧边栏：评估模块 (Phase 6) ---
with st.sidebar:
    st.header("📊 Phase 6: Evaluation")
    st.markdown("Benchmark generated classes against a Ground Truth list.")
    
    ground_truth_str = st.text_area("Expected Classes (comma separated):", 
                                    value="BankSystem, Customer, Account, Administrator",
                                    help="Enter the list of classes that SHOULD be detected.")
    
    if st.button("Calculate Metrics"):
        if not system.classes:
            st.warning("Please generate a diagram first.")
        else:
            expected = set([x.strip() for x in ground_truth_str.split(",") if x.strip()])
            detected = set(system.classes.keys())
            
            # 计算指标
            tp = len(expected.intersection(detected))
            fp = len(detected - expected)
            fn = len(expected - detected)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("Precision", f"{precision:.2f}")
            c2.metric("Recall", f"{recall:.2f}")
            c3.metric("F1-Score", f"{f1:.2f}")
            
            if fp > 0: st.warning(f"False Positives (Extra): {', '.join(detected - expected)}")
            if fn > 0: st.error(f"False Negatives (Missed): {', '.join(expected - detected)}")

# --- 主界面 ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Requirements Input")
    default_text = """The BankSystem allows a Customer to open an Account.
The Account is managed by the Administrator. 
The Customer has many Orders.
A SavingsAccount is a type of Account.
The Customer places an Order."""
    
    user_input = st.text_area("Enter text:", value=default_text, height=250)
    
    generate_btn = st.button("Generate Diagram", type="primary")

if generate_btn and user_input:
    with st.spinner("Analyzing semantics, structure & multiplicity..."):
        uml_code = system.process(user_input)
    
    # 显示结果
    with col1:
        if system.ontology_suggestions:
            with st.expander("💡 Ontology Suggestions (Phase 4)", expanded=True):
                for s in system.ontology_suggestions:
                    st.caption(s)
        
        with st.expander("View PlantUML Code"):
            st.code(uml_code, language='java')

    with col2:
        st.subheader("📊 Class Diagram")
        try:
            server = PlantUML(url='http://www.plantuml.com/plantuml/img/')
            image_url = server.get_url(uml_code)
            st.image(image_url, caption="Generated Diagram")
        except Exception as e:
            st.error(f"Visualization Error: {e}")
            
