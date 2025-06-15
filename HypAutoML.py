import streamlit as st
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import json
import re
from fpdf import FPDF
from groq import Groq

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import shap
import lime
import lime.lime_tabular


# -----------------------------------
# Prompts, JSON Parsing, Groq Handling
# -----------------------------------
class PromptHandler:
    """Handles all interactions with Groq API and prompt engineering"""

    def __init__(self, api_key):
        # Initialize Groq client with API key
        self.client = Groq(api_key=api_key)

    def call_groq(self, prompt):
        """Sends prompt to Groq API and returns response"""
        return (
            self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-70b-8192",
                stream=False,
            )
            .choices[0]
            .message.content
        )

    def extract_first_json_array(self, text):
        """Extracts and validates first JSON array from text response"""
        # Use regex to find JSON blocks
        json_blocks = re.findall(r"\[\s*{.*?}\s*]", text, re.DOTALL)
        for block in json_blocks:
            # Handle scientific notation in JSON
            block = re.sub(
                r"10\\^(-?\\d+)", lambda m: str(10 ** int(m.group(1))), block
            )
            try:
                # Parse JSON and validate structure
                features = json.loads(block)
                required_keys = {
                    "name",
                    "description",
                    "analogy_to_real_world",
                    "analogy_description",
                    "range_min",
                    "range_max",
                    "units",
                }
                # Check if all required keys are present
                if all(
                    isinstance(item, dict) and required_keys.issubset(item.keys())
                    for item in features
                ):
                    return features
            except json.JSONDecodeError:
                continue  # Skip invalid JSON

    def generate_features_from_hypothesis(self, hypothesis, mode):
        """Generates feature definitions based on hypothesis and mode"""
        # Define prompt variations based on selected mode
        realism_prompt = {
            "Scientific Realism": "Only generate measurable features using physical science.",
            "Mirror World Magic": (
                "You are a magical AI in a mirrored Earth. Return a JSON array of 10 realistic features."
                " Each must include: name, description, analogy_to_real_world, analogy_description,"
                " range_min, range_max, units, distribution_type (normal/lognormal/uniform), mean and std if applicable."
            ),
            "Analogical Simulation": "Use creative and plausible analogies.",
        }.get(mode, "Use creative and plausible analogies.")

        # Construct the prompt
        prompt = f"""
        Hypothesis: '{hypothesis}'
        Generate exactly 10 features as a JSON array with:
        - name, description, analogy_to_real_world, analogy_description
        - range_min, range_max, units
        - distribution_type (normal, lognormal, uniform)
        - mean and std if available
        {realism_prompt}
        Return only the JSON array.
        """

        # Get response from Groq
        content = self.call_groq(prompt)
        try:
            return self.extract_first_json_array(content)
        except:
            # Attempt to fix malformed JSON
            fix_prompt = f"The following JSON is malformed. Please fix it and return only JSON:\n{content}"
            return self.extract_first_json_array(self.call_groq(fix_prompt))

    def generate_logical_rule(self, hypothesis, feature_names):
        """Generates a classification rule using features"""
        prompt = f"""
        You are a top-tier AI scientist. Create a Python rule using 5 features to classify:
        Hypothesis: {hypothesis}
        Use: {', '.join(feature_names)}
        Format:
        ```python
        if (conditions):
            return 1
        else:
            return 0
        ```
        """
        # Get rule from Groq and extract the code block
        response = self.call_groq(prompt)
        match = re.search(r"if .*", response, re.DOTALL)
        return match.group(0).strip() if match else "if True: return 1 else: return 0"

    def generate_reasoning_report(self, hypothesis, feature_names, df):
        """Generates analysis report based on synthetic data"""
        # Create sample data for the report
        sample_json = df.head(20).to_json(orient="records")
        features_str = ", ".join(feature_names)

        # Construct analysis prompt
        prompt = f"""
        Analyze this synthetic dataset for hypothesis: \"{hypothesis}\"
        Features: {features_str}
        Sample: {sample_json}

        Provide:
        1. Basic insights
        2. SHAP-based validation
        3. Contrapositive logic
        4. Simulated scientist voting
        5. Appendix on interpretability
        """
        return self.call_groq(prompt)


# -----------------
# Plotting Utilities
# -----------------
class PlotUtils:
    """Handles all visualization components"""

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred):
        """Plots confusion matrix"""
        fig, ax = plt.subplots()
        sns.heatmap(
            confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    @staticmethod
    def plot_roc_curve(y_true, y_scores):
        """Plots ROC curve with AUC score"""
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"ROC AUC = {auc(fpr, tpr):.2f}")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend()
        st.pyplot(fig)

    @staticmethod
    def apply_kmeans_and_plot(df, n_clusters=2):
        """Applies K-Means clustering and visualizes results"""
        st.subheader("🔍 K-Means Clustering Analysis")
        # Prepare data for clustering
        X_cluster = df.drop("target", axis=1)
        X_cluster = SimpleImputer().fit_transform(X_cluster)
        X_scaled = StandardScaler().fit_transform(X_cluster)

        # Apply K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        df["cluster"] = clusters

        # Calculate and display silhouette score
        score = silhouette_score(X_scaled, clusters)
        st.metric("Silhouette Score", f"{score:.3f}")

        # Reduce dimensions for visualization
        pca = PCA(n_components=2)
        components = pca.fit_transform(X_scaled)

        # Create cluster visualization
        fig, ax = plt.subplots()
        ax.scatter(
            components[:, 0], components[:, 1], c=clusters, cmap="viridis", alpha=0.6
        )
        ax.set_title("K-Means Clustering (PCA)")
        st.pyplot(fig)
        return df


# ----------------------
# Model Training & Explain
# ----------------------
class ModelHandler:
    """Handles model training and data generation"""

    @staticmethod
    def safe_eval_logic(row, logic_code):
        """Safely evaluates classification rule on a data row"""
        local_vars = row.to_dict()
        try:
            # Dynamically execute the classification rule
            exec(
                f"def rule_fn():\n    {logic_code.replace('return', 'result =')}\n    return result",
                globals(),
                local_vars,
            )
            return int(local_vars["rule_fn"]())
        except Exception:
            # Fallback to random classification on error
            return random.randint(0, 1)

    @staticmethod
    def generate_features_with_correlation(feature_defs, correlations, n_samples):
        """Generates synthetic data with user-defined feature correlations"""
        # Map correlation strength to numerical values
        strength_map = {"High": 0.9, "Medium": 0.5, "Low": 0.1}

        # Create base signal for correlations
        target_base = np.random.rand(n_samples)
        df = pd.DataFrame()

        # Generate each feature based on its definition
        for feature in feature_defs:
            name = feature["name"]
            # Get correlation strength for this feature
            strength = strength_map[correlations.get(name, "Medium")]

            # Create correlated signal with noise
            noise = np.random.normal(0, (1 - strength), n_samples)
            signal = target_base * strength + noise

            # Scale to specified range
            values = (signal - signal.min()) / (signal.max() - signal.min())
            scaled = feature["range_min"] + values * (
                feature["range_max"] - feature["range_min"]
            )
            df[name] = np.clip(scaled, feature["range_min"], feature["range_max"])
        return df

    @staticmethod
    def generate_pdf_report(content):
        """Generates PDF from analysis report content"""
        if not content.strip():
            return None
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Arial", size=10)

        # Add content to PDF
        for line in content.split("\n"):
            pdf.multi_cell(0, 10, line)

        # Save PDF file
        path = "groq_report.pdf"
        pdf.output(path)
        return path


# --------------------
# Streamlit UI Handler
# --------------------
class HypothesisAutoMLApp:
    """Main application class for Streamlit UI"""

    def __init__(self):
        # Initialize components with Groq API key
        self.prompter = PromptHandler(
            api_key="**********************************************************"
        )
        self.plotter = PlotUtils()
        self.modeler = ModelHandler()

    def run(self):
        """Main application runner"""
        # Configure Streamlit page
        st.set_page_config(page_title="Hypothesis AutoML", layout="wide")
        st.title("🔮 Hypothesis-Driven AutoML Simulator")
        st.markdown("## 📘 How Feature Generation Works")
        st.markdown("This app simulates features as if your hypothesis is true.")

        # User inputs
        hypothesis = st.text_input(
            "🧠 Enter Hypothesis", "IAS success depends on study and mocks"
        )
        n_samples = st.slider("Samples", 100, 5000, 1000)
        task_type = st.selectbox("Task", ["Classification", "Regression"])
        mode = st.selectbox(
            "Mode",
            ["Analogical Simulation", "Scientific Realism", "Mirror World Magic"],
        )
        include_noise = st.checkbox("Add noise/edge cases", True)

        # Feature generation
        if "feature_defs" not in st.session_state and st.button(
            "🔮 Generate 20 Parameters"
        ):
            st.session_state.feature_defs = (
                self.prompter.generate_features_from_hypothesis(hypothesis, mode)
            )

        # Data generation and modeling
        if "feature_defs" in st.session_state:
            st.markdown("### 🔹 Correlation Strength (User Defined)")
            correlations = {}

            # Display correlation controls for each feature
            for f in st.session_state.feature_defs:
                scale = st.radio(
                    f"{f['name']} ({f['description']})",
                    ["High", "Medium", "Low"],
                    horizontal=True,
                    key=f"cor_{f['name']}",
                )
                correlations[f["name"]] = scale

            if st.button("✅ Confirm Correlation Preferences & Generate Data"):
                features = st.session_state.feature_defs
                feature_names = [f["name"] for f in features]

                # Generate synthetic data
                df = self.modeler.generate_features_with_correlation(
                    features, correlations, n_samples
                )

                # Add noise/edge cases if requested
                if include_noise:
                    # Add outliers
                    for col in df.columns[:1]:
                        df.loc[df.sample(frac=0.05).index, col] *= random.choice(
                            [10, 0.1]
                        )
                    # Add missing values
                    for col in df.columns:
                        df.loc[df.sample(frac=0.05).index, col] = np.nan

                # Generate classification rule
                logic_code = self.prompter.generate_logical_rule(
                    hypothesis, feature_names
                )
                # Apply rule to create target column
                df["target"] = df.apply(
                    lambda row: self.modeler.safe_eval_logic(row, logic_code), axis=1
                )
                st.code(logic_code)

                # Apply clustering and visualize
                df = self.plotter.apply_kmeans_and_plot(df)

                # Prepare data for modeling
                X = df.drop(["target", "cluster"], axis=1)
                y = df["target"]
                X = SimpleImputer().fit_transform(X)
                X = StandardScaler().fit_transform(X)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

                # Model selection based on task type
                models = (
                    {
                        "Logistic Regression": LogisticRegression(),
                        "Random Forest": RandomForestClassifier(),
                        "Decision Tree": DecisionTreeClassifier(),
                        "Neural Net": MLPClassifier(max_iter=500),
                        "Naive Bayes": GaussianNB(),
                        "AdaBoost": AdaBoostClassifier(),
                        "SVM": SVC(probability=True),
                    }
                    if task_type == "Classification"
                    else {"Linear Regression": LinearRegression()}
                )

                # Train and evaluate each model
                for name, model in models.items():
                    st.subheader(name)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # Evaluation metrics
                    if task_type == "Classification":
                        st.code(classification_report(y_test, y_pred))
                        try:
                            # Plot ROC curve if possible
                            y_score = model.predict_proba(X_test)[:, 1]
                            self.plotter.plot_roc_curve(y_test, y_score)
                        except:
                            st.warning("ROC not available")
                        self.plotter.plot_confusion_matrix(y_test, y_pred)
                    else:
                        # Regression metrics
                        st.markdown(
                            f"**MSE:** {mean_squared_error(y_test, y_pred):.2f}"
                        )
                        st.markdown(f"**R2:** {r2_score(y_test, y_pred):.2f}")

                    # SHAP explainability
                    try:
                        explainer = shap.Explainer(model, X_train)
                        shap_values = explainer(X_test[:100])
                        st.subheader("SHAP Summary")
                        fig, ax = plt.subplots()
                        shap.plots.beeswarm(shap_values, show=False)
                        st.pyplot(fig)
                    except:
                        st.warning("SHAP failed")

                    # LIME explainability
                    try:
                        lime_exp = lime.lime_tabular.LimeTabularExplainer(
                            X_train,
                            feature_names=df.drop(
                                ["target", "cluster"], axis=1
                            ).columns,
                            class_names=(
                                ["0", "1"] if task_type == "Classification" else []
                            ),
                            mode=(
                                "classification"
                                if task_type == "Classification"
                                else "regression"
                            ),
                        )
                        explanation = lime_exp.explain_instance(
                            X_test[0],
                            (
                                model.predict_proba
                                if task_type == "Classification"
                                else model.predict
                            ),
                            num_features=5,
                        )
                        st.subheader("LIME Explanation")
                        st.text(explanation.as_list())
                    except:
                        st.warning("LIME failed")

                # Generate AI reasoning report
                st.header("📊 AI Reasoning Report")
                with st.spinner("Groq generating insights..."):
                    report = self.prompter.generate_reasoning_report(
                        hypothesis, feature_names, df
                    )
                    st.markdown(report)
                    path = self.modeler.generate_pdf_report(report)
                    if path:
                        with open(path, "rb") as f:
                            st.download_button(
                                "Download Report as PDF", f, file_name="AI_Report.pdf"
                            )
                st.caption(
                    "Built by Aditya | Groq + SHAP + LIME + Clustering + Scaling + Correlation UI"
                )


if __name__ == "__main__":
    app = HypothesisAutoMLApp()
    app.run()
