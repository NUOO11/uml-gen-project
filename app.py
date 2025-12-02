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
# 1. System Configuration & Resource Loader
#    系统配置与资源加载 (稳健模式)
# ==========================================
st.set_page_config(
    page_title="NLP to UML: Intelligent Generator",
    page_icon="🧬",
    layout="wide"
)

@st.cache_resource
def load_resources():
    """
    Load NLP resources manually to bypass Streamlit Cloud permission issues.
    手动加载资源，绕过云端权限限制。
    """
    # 1. Load NLTK (WordNet for Phase 4: Semantics)
    try:
        nltk.data.find('corpora/wordnet.zip')
    except LookupError:
        nltk.download('wordnet')
        nltk.download('omw-1.4')

    # 2. Download and Load Spacy Model (Phase 2: NLP Pipeline)
    MODEL_URL = "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1.tar.gz"
    EXTRACT_PATH = "./model_data"
    MODEL_PATH = f"{EXTRACT_PATH}/en_core_web_sm-3.7.1/en_core_web_sm/en_core_web_sm-3.7.1"
    
    # Check if model exists locally
    if not os.path.exists(MODEL_PATH):
        try:
            if not os.path.exists(EXTRACT_PATH):
                os.makedirs(EXTRACT_PATH)
            
            # Download model file
            response = requests.get(MODEL_URL, stream=True)
            if response.status_code == 200:
                with open("model.tar.gz", 'wb') as f:
                    f.write(response.raw.read())
                # Extract
                with tarfile.open("model.tar.gz", "r:gz") as tar:
                    tar.extractall(path=EXTRACT_PATH)
            else:
                st.error("Failed to download NLP model. Please check network.")
                st.stop()
        except Exception as e:
            st.error(f"Error initializing NLP engine: {e}")
            st.stop()
    
    # Load the model
    try:
        return spacy.load(MODEL_PATH)
    except OSError:
        # Fallback for local environments
        import en_core_web_sm
        return en_core_web_sm.load()

# Initialize System
with st.spinner("🚀 Initializing NLP Pipeline & Ontology Database..."):
    nlp = load_resources()

# ==========================================
# 2. Core Logic: Hybrid Extraction System
#    核心逻辑：混合提取系统 (对应 Phase 3 & 4)
# ==========================================

class HybridUMLSystem:
    def __init__(self):
        self.classes = {}  # Store classes and their members
        self.relationships = [] # Store relationships (Source, Type, Target, Label)
        self.ontology_suggestions = [] # Store Phase 4 suggestions
        # Verbs to ignore (linking verbs)
        self.ignored_verbs = {"be", "have", "include", "consist", "contain", "involve"}

    def _get_lemma(self, token):
        return token.lemma_.lower()

    def check_ontology_entity(self, word):
        """
        Phase 4: Semantic Analysis using WordNet.
        Check if a word is a valid physical or abstract entity (Noun).
        """
        try:
            synsets = wordnet.synsets(word)
            if not synsets: return True # Keep unknown words as potential classes
            return any(s.pos() == 'n' for s in synsets)
        except:
            return True

    def suggest_parent_hierarchy(self, class_name):
        """
        Phase 4 Enhancement: Ontology-based Hierarchy Suggestion.
        Infer implicit inheritance (e.g., Car -> Vehicle).
        """
        try:
            syn = wordnet.synsets(class_name.lower())
            if syn:
                hypernyms = syn[0].hypernyms()
                if hypernyms:
                    parent = hypernyms[0].lemmas()[0].name().capitalize()
                    # Filter out too generic terms
                    if parent not in ["Entity", "Object", "Whole", "Physical_entity", "Abstraction"]:
                        return parent
        except:
            pass
        return None

    def detect_multiplicity(self, token):
        """
        Phase 3: Multiplicity Detection (1..*).
        Detect keywords like 'many', 'list of', or plural nouns.
        """
        for child in token.children:
            if child.text.lower() in ["many", "multiple", "list", "set", "collection", "all", "various"]:
                return "1..*"
            if child.tag_ == "NNS": # Plural noun tag
                return "0..*"
        return "1"

    def process_requirements(self, text):
        """
        Main Pipeline: Preprocessing -> Extraction -> Generation
        """
        # Reset state
        self.classes = {}
        self.relationships = []
        self.ontology_suggestions = []
        
        # NLP Pipeline (Phase 2)
        doc = nlp(text)
        
        # --- Pass 1: Entity Extraction (Identifying Classes) ---
        for token in doc:
            # Rule: Noun Subjects/Objects are potential Classes
            if token.pos_ in ["NOUN", "PROPN"] and token.dep_ in ["nsubj", "dobj", "pobj", "nsubjpass"]:
                if self.check_ontology_entity(token.lemma_):
                    class_name = token.lemma_.capitalize()
                    if class_name not in self.classes:
                        self.classes[class_name] = {'attributes': set(), 'methods': set()}
                        
                        # Ontology Hint
                        parent = self.suggest_parent_hierarchy(class_name)
                        if parent:
                            self.ontology_suggestions.append(f"💡 Ontology Hint: '{class_name}' is a type of '{parent}'")

        # --- Pass 2: Relationship & Feature Extraction ---
        for token in doc:
            
            # 1. Inheritance ("Is-a" relationship)
            if token.lemma_ == "be":
                subjects = [c for c in token.children if c.dep_ == "nsubj"]
                attrs = [c for c in token.children if c.dep_ == "attr"]
                if subjects and attrs:
                    child = subjects[0].lemma_.capitalize()
                    parent = attrs[0].lemma_.capitalize()
                    if child in self.classes and parent in self.classes:
                        self.relationships.append((child, "<|--", parent, ""))

            # 2. Aggregation & Attributes ("Has-a" relationship)
            elif token.lemma_ in ["have", "contain", "include", "own"]:
                owners = [c for c in token.children if c.dep_ == "nsubj"]
                objs = [c for c in token.children if c.dep_ == "dobj"]
                if owners and objs:
                    owner_name = owners[0].lemma_.capitalize()
                    # Check Multiplicity
                    mult_val = self.detect_multiplicity(objs[0])
                    mult_label = f'"{mult_val}"' if mult_val != "1" else ""
                    
                    if owner_name in self.classes:
                        obj_lemma = objs[0].lemma_.capitalize()
                        # If object is a Class -> Aggregation
                        if obj_lemma in self.classes:
                            self.relationships.append((owner_name, "o--", obj_lemma, mult_label))
                        else:
                            # If object is not a Class -> Attribute
                            self.classes[owner_name]['attributes'].add(objs[0].text)

            # 3. Association & Methods (Active Verbs)
            elif token.pos_ == "VERB" and token.lemma_ not in self.ignored_verbs:
                subjects = [c for c in token.children if c.dep_ == "nsubj"]
                if subjects:
                    subj_name = subjects[0].lemma_.capitalize()
                    if subj_name in self.classes:
                        # Add as Method
                        self.classes[subj_name]['methods'].add(token.lemma_)
                        
                        # Check for Association (Subject -> Object)
                        objects = [c for c in token.children if c.dep_ == "dobj"]
                        if objects:
                            obj_name = objects[0].lemma_.capitalize()
                            if obj_name in self.classes and subj_name != obj_name:
                                self.relationships.append((subj_name, "-->", obj_name, f": {token.lemma_}"))

            # 4. Passive Voice Handling (Enhancement)
            # e.g., "The Account is managed by the Administrator."
            if token.dep_ == "agent" and token.head.pos_ == "VERB":
                # token is 'by', head is 'managed'
                actual_subjects = [c for c in token.children if c.dep_ == "pobj"] # Administrator
                verb = token.head
                passive_subjects = [c for c in verb.children if c.dep_ == "nsubjpass"] # Account
                
                if actual_subjects and passive_subjects:
                    actor = actual_subjects[0].lemma_.capitalize()
                    receiver = passive_subjects[0].lemma_.capitalize()
                    method = verb.lemma_
                    
                    # Ensure classes exist
                    for name in [actor, receiver]:
                        if name not in self.classes:
                            self.classes[name] = {'attributes': set(), 'methods': set()}
                    
                    self.classes[actor]['methods'].add(method)
                    self.relationships.append((actor, "-->", receiver, f": {method}"))

        return self.generate_plantuml()

    def generate_plantuml(self):
        """Phase 5: Generation"""
        lines = ["@startuml", "skinparam classAttributeIconSize 0", "hide circle", "skinparam shadowing false"]
        
        # Classes
        for cls_name, details in self.classes.items():
            lines.append(f"class {cls_name} {{")
            for attr in details['attributes']:
                lines.append(f"  - {attr}")
            if details['attributes'] and details['methods']: lines.append("  ..")
            for method in details['methods']:
                lines.append(f"  + {method}()")
            lines.append("}")
            
        # Relationships
        for src, rel_type, target, label in set(self.relationships):
            lines.append(f"{src} {rel_type} {target} {label}")
            
        lines.append("@enduml")
        return "\n".join(lines)

system = HybridUMLSystem()

# ==========================================
# 3. User Interface (English)
#    用户界面 (全英文)
# ==========================================

st.title("🎓 NLP to UML Generation System")
st.markdown("""
**Project Title:** Generate UML Class Diagram from Text Specification Using Natural Language Processing  
**Methodology:** Hybrid Approach (Rule-based Extraction + Semantic Analysis)
""")

# --- Sidebar: Phase 6 Evaluation ---
with st.sidebar:
    st.header("📊 Phase 6: Evaluation")
    st.info("Benchmark the system output against Ground Truth (Expert Logic).")
    
    ground_truth_input = st.text_area("Expected Classes (Ground Truth):", 
                                      value="BankSystem, Customer, Account, Administrator",
                                      help="Enter the comma-separated list of classes that MUST be detected.")
    
    if st.button("Calculate Metrics"):
        if not system.classes:
            st.warning("Please generate a diagram first.")
        else:
            expected = set([x.strip() for x in ground_truth_input.split(",") if x.strip()])
            detected = set(system.classes.keys())
            
            # Calculation
            tp = len(expected.intersection(detected))
            fp = len(detected - expected)
            fn = len(expected - detected)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            st.markdown("### Results")
            c1, c2, c3 = st.columns(3)
            c1.metric("Precision", f"{precision:.2f}")
            c2.metric("Recall", f"{recall:.2f}")
            c3.metric("F1-Score", f"{f1:.2f}")
            
            if fp > 0: st.warning(f"False Positives (Extra): {', '.join(detected - expected)}")
            if fn > 0: st.error(f"False Negatives (Missed): {', '.join(expected - detected)}")

# --- Main Interface ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Phase 1: Data Input")
    st.markdown("Enter your **Software Requirements Specification (SRS)** below:")
    
    default_text = """The BankSystem allows a Customer to open an Account.
The Account is managed by the Administrator. 
The Customer has many Orders.
The Customer places an Order.
A SavingsAccount is a type of Account."""
    
    user_input = st.text_area("Requirement Text:", value=default_text, height=300)
    
    generate_btn = st.button("Generate Diagram", type="primary")

if generate_btn and user_input:
    with st.spinner("Processing (Phase 2 & 3)... Analyzing Dependency & Semantics..."):
        uml_code = system.process_requirements(user_input)
    
    # Left Column: Code & Suggestions
    with col1:
        if system.ontology_suggestions:
            with st.expander("💡 Phase 4: Semantic & Ontology Hints", expanded=True):
                for s in system.ontology_suggestions:
                    st.caption(s)
        
        st.subheader("💻 Generated PlantUML Code")
        st.code(uml_code, language='java')

    # Right Column: Visualization
    with col2:
        st.subheader("📊 Phase 5: Visualization")
        try:
            # Force HTTPS for browser security
            server = PlantUML(url='https://www.plantuml.com/plantuml/img/')
            image_url = server.get_url(uml_code)
            
            st.image(image_url, caption="Generated Class Diagram")
            st.markdown(f"**[🔗 Click here if image does not load]({image_url})**")
            
        except Exception as e:
            st.error("Visualization Service Unavailable.")
            st.markdown(f"[View on PlantText](https://www.planttext.com/?text={uml_code})")
