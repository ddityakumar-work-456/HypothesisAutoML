# -----------------------------------------------------
# Hypothesis AutoML – Research Enhanced Full App (Final)
# -----------------------------------------------------
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
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)

    def call_groq(self, prompt):
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
        json_blocks = re.findall(r"\[\s*{.*?}\s*]", text, re.DOTALL)
        for block in json_blocks:
            block = re.sub(
                r"10\\\\\^(-?\\\\\d+)", lambda m: str(10 ** int(m.group(1))), block
            )
            try:
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
                if all(
                    isinstance(item, dict) and required_keys.issubset(item.keys())
                    for item in features
                ):
                    return features
            except json.JSONDecodeError:
                continue

    def generate_features_from_hypothesis(self, hypothesis, mode):
        realism_prompt = {
            "Scientific Realism": "Only generate measurable features using physical science.",
            "Mirror World Magic": (
                "You are a magical AI in a mirrored Earth. Return a JSON array of 10 realistic features."
                " Each must include: name, description, analogy_to_real_world, analogy_description,"
                " range_min, range_max, units, distribution_type (normal/lognormal/uniform), mean and std if applicable."
            ),
            "Analogical Simulation": "Use creative and plausible analogies.",
        }.get(mode, "Use creative and plausible analogies.")

        prompt = f"""
        Hypothesis: '{hypothesis}'
        Generate exactly 20 features as a JSON array with:
        - name, description, analogy_to_real_world, analogy_description
        - range_min, range_max, units
        - distribution_type (normal, lognormal, uniform)
        - mean and std if available
        {realism_prompt}
        Return only the JSON array.
        """

        content = self.call_groq(prompt)

        # print(content)
        features = self.extract_first_json_array(content)

        if not features:
            fix_prompt = f"The following JSON is malformed or empty. Please fix it and return only a valid JSON array:\n{content}"
            fixed_content = self.call_groq(fix_prompt)
            features = self.extract_first_json_array(fixed_content)

        return features

    def generate_logical_rule(self, hypothesis, feature_names):
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
        response = self.call_groq(prompt)
        match = re.search(r"if .*", response, re.DOTALL)
        return match.group(0).strip() if match else "if True: return 1 else: return 0"

    def generate_reasoning_report(self, hypothesis, feature_names, df):
        sample_json = df.head(20).to_json(orient="records")
        features_str = ", ".join(feature_names)
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

    def generate_flow_chart(self, hypothesis):
        prompt = f"""
        Assume you are a top-level scientist, engineer, and systems architect with
        unlimited access to advanced materials, energy sources, AI, quantum 
        computers, and infrastructure. Your mission is to achieve the following 
        hypothesis through physical construction and implementation alone — not 
        simulations, theories, or research.

        🔬 Hypothesis:
        {hypothesis}

        Create a wireframe execution blueprint that covers how to achieve
        this hypothesis in the real world using physical machines, infrastructure, and systems.

        Do not include R&D, simulations, theoretical proofs, or 
        validation discussions. Focus only on designing and building what’s 
        needed to make it real.

        🧩 Your Plan Must Include:
        ✅ Single Best Approach Chosen – from all possible methods

        🏗️ Master Machine/Component List – what must be physically built

        🔃 Chronological Construction Order – step-by-step build phases

        ⚙️ Purpose of Each Machine – what it does and how it connects

        🔑 Trigger Mechanism & Operation – how the hypothesis is activated

        🛡️ Safety, Control, and AI Systems – to prevent failure or danger

        📋 Final Deployment Checklist – confirm readiness and launch

        📚 Appendix – brief explanation of all complex terms used
        """
        return self.call_groq(prompt)

    def generate_theory_snippets(self, hypothesis):
        prompt = f"""
        Hypothesis: {hypothesis}
        Provide 3 short scientific theories or concepts that relate to or support this hypothesis. Each must be concise (max 2 lines).
        """
        return self.call_groq(prompt)

    def suggest_scientific_experiments(self, hypothesis):
        prompt = f"""
        Hypothesis: {hypothesis}
        Suggest 2-3 related scientific experiments (real or simulated) that can help test or explore this hypothesis. Be creative and concise.
        """
        return self.call_groq(prompt)

    def generate_concept_notes(self, hypothesis):
        prompt = f"""
        Hypothesis: {hypothesis}
        Provide foundational concept notes (max 5 bullet points) to help a non-expert understand the underlying science or logic behind the hypothesis.
        """
        return self.call_groq(prompt)


# --------------------------------------------
# Model Training and Reporting
# --------------------------------------------
class ModelHandler:
    def safe_eval_logic(self, row, logic_code):
        local_vars = row.to_dict()
        try:
            exec(
                f"def rule_fn():\n    {logic_code.replace('return', 'result =')}\n    return result",
                globals(),
                local_vars,
            )
            return int(local_vars["rule_fn"]())
        except Exception:
            return random.randint(0, 1)

    # def generate_features_with_correlation(self, feature_defs, correlations, n_samples):
    #     strength_map = {"High": 0.9, "Medium": 0.5, "Low": 0.1}
    #     target_base = np.random.rand(n_samples)
    #     df = pd.DataFrame()
    #     for feature in feature_defs:
    #         name = feature["name"]
    #         strength = strength_map[correlations.get(name, "Medium")]
    #         noise = np.random.normal(0, (1 - strength), n_samples)
    #         signal = target_base * strength + noise
    #         values = (signal - signal.min()) / (signal.max() - signal.min())
    #         scaled = feature["range_min"] + values * (
    #             feature["range_max"] - feature["range_min"]
    #         )
    #         df[name] = np.clip(scaled, feature["range_min"], feature["range_max"])
    #     return df

    def generate_features_with_correlation(
        self,
        feature_defs,
        correlations,
        n_samples,
        generate_target=True,
        flip_ratio=0.2,
    ):
        """
        Generates features with specified correlation to a synthetic target.
        Optionally generates a binary target with noise for training ML models.

        Parameters:
            feature_defs (list): List of dicts with 'name', 'range_min', 'range_max'.
            correlations (dict): Feature-to-correlation strength mapping.
            n_samples (int): Number of data samples.
            generate_target (bool): Whether to generate synthetic binary target.
            flip_ratio (float): % of target labels to flip (for noise simulation).

        Returns:
            pd.DataFrame: DataFrame with features (and target if generate_target is True).
        """

        strength_map = {"High": 0.9, "Medium": 0.5, "Low": 0.1}
        target_base = np.random.rand(n_samples)
        df = pd.DataFrame()

        feature_weights = {}
        for feature in feature_defs:
            name = feature["name"]
            strength = strength_map.get(correlations.get(name, "Medium"), 0.5)
            noise = np.random.normal(0, (1 - strength), n_samples)
            signal = target_base * strength + noise
            values = (signal - signal.min()) / (signal.max() - signal.min())
            scaled = feature["range_min"] + values * (
                feature["range_max"] - feature["range_min"]
            )
            df[name] = np.clip(scaled, feature["range_min"], feature["range_max"])
            feature_weights[name] = np.round(
                np.random.uniform(-2, 2), 2
            )  # store random weight

        if generate_target:
            # Step 1: Generate target using logistic model
            weights = [feature_weights[name] for name in df.columns]
            w0 = np.random.uniform(-1, 1)  # bias
            logit = w0 + np.dot(df.values, np.array(weights))
            prob = 1 / (1 + np.exp(-logit))
            df["target"] = np.random.binomial(1, prob)

            # Step 2: Flip % of each class to simulate label noise
            noisy_df = pd.DataFrame()
            for label in df["target"].unique():
                subset = df[df["target"] == label].copy()
                flip_count = int(flip_ratio * len(subset))
                if flip_count > 0:
                    flip_indices = subset.sample(flip_count, random_state=42).index
                    subset.loc[flip_indices, "target"] = 1 - label
                noisy_df = pd.concat([noisy_df, subset], axis=0)

            df = noisy_df.sample(frac=1, random_state=42).reset_index(drop=True)

        return df

    def generate_pdf_report(self, content):
        if not content.strip():
            return None
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Arial", size=10)
        for line in content.split("\n"):
            pdf.multi_cell(0, 10, line)
        path = "groq_report.pdf"
        pdf.output(path)
        return path


# --------------------------------------------
# Full Application Logic
# --------------------------------------------
class HypothesisAutoMLApp:
    def __init__(self):
        self.prompter = PromptHandler(
            api_key="***************************************************"
        )
        self.modeler = ModelHandler()

    def run(self):
        st.set_page_config(
            page_title="Hypothesis AutoML - Research Tool", layout="wide"
        )
        st.title("🔬 Hypothesis Research & Modeling Simulator")

        hypothesis = st.text_input("🧠 Enter Your Hypothesis")
        if hypothesis:
            st.markdown("---")
            st.subheader("Solving the Unsolvable: A Step-By-Step Approach")
            st.markdown(self.prompter.generate_flow_chart(hypothesis))

            st.subheader("📖 Scientific Theories")
            st.markdown(self.prompter.generate_theory_snippets(hypothesis))

            # st.subheader("🔬 Suggested Experiments")
            # st.markdown(self.prompter.suggest_scientific_experiments(hypothesis))

            st.subheader("📒 Base Concept Notes")
            st.markdown(self.prompter.generate_concept_notes(hypothesis))

        if st.button("🔮 Generate Feature Candidates") and hypothesis:

            st.session_state.feature_defs = (
                self.prompter.generate_features_from_hypothesis(
                    hypothesis, mode="Scientific Realism"
                )
            )

        if "feature_defs" in st.session_state:
            all_features = st.session_state.feature_defs
            selected = st.multiselect(
                "✅ Choose Features to Use for Modeling",
                options=[
                    f["name"] + "| [" + f["description"] + "]" for f in all_features
                ],
            )

            # print(selected)
            selected_defs = [
                f
                for f in all_features
                if f["name"] in [y.split("|")[0] for y in selected]
            ]

            # print(selected_defs)

            correlations = {}
            for feat in selected_defs:
                scale = st.radio(
                    f"{feat['name']} ({feat['description']})",
                    ["High", "Medium", "Low"],
                    horizontal=True,
                    key=f"cor_{feat['name']}",
                )
                correlations[feat["name"]] = scale

            if st.button("🚀 Run Modeling"):
                df = self.modeler.generate_features_with_correlation(
                    selected_defs, correlations, n_samples=1000
                )
                logic_code = self.prompter.generate_logical_rule(
                    hypothesis, [f["name"] for f in selected_defs]
                )
                df["target"] = df.apply(
                    lambda row: self.modeler.safe_eval_logic(row, logic_code), axis=1
                )

                # Step 2: Flip 20% of labels per class (binary or multiclass)
                noisy_df = pd.DataFrame()
                all_labels = df["target"].unique().tolist()

                for label in all_labels:
                    subset = df[df["target"] == label].copy()
                    flip_count = int(0.3 * len(subset))

                    if flip_count > 0:
                        flip_indices = subset.sample(flip_count, random_state=42).index
                        remaining_labels = [l for l in all_labels if l != label]
                        subset.loc[flip_indices, "target"] = np.random.choice(
                            remaining_labels, size=flip_count
                        )

                    noisy_df = pd.concat([noisy_df, subset], axis=0)

                # Optional shuffle
                df = noisy_df.sample(frac=1, random_state=42).reset_index(drop=True)

                # df.to_csv("fyt.csv", index=False)
                X = df.drop("target", axis=1)
                y = df["target"]

                st.code(logic_code)
                st.markdown("### 🔍 Clustering with PCA")
                X_imp = SimpleImputer().fit_transform(X)
                X_scaled = StandardScaler().fit_transform(X_imp)
                # kmeans = KMeans(n_clusters=2, random_state=42)
                # df["cluster"] = kmeans.fit_predict(X_scaled)
                # pca = PCA(n_components=2)
                # pca_res = pca.fit_transform(X_scaled)
                # fig, ax = plt.subplots()
                # ax.scatter(
                #     pca_res[:, 0],
                #     pca_res[:, 1],
                #     c=df["cluster"],
                #     cmap="viridis",
                #     alpha=0.6,
                # )
                # ax.set_title("Clustering Visualization")
                # st.pyplot(fig)

                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.3
                )
                models = {
                    "Logistic Regression": LogisticRegression(),
                    "Random Forest": RandomForestClassifier(),
                    "Neural Net": MLPClassifier(max_iter=500),
                }

                for name, model in models.items():
                    st.subheader(name)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    st.code(classification_report(y_test, y_pred))

                    try:
                        y_score = model.predict_proba(X_test)[:, 1]
                        fpr, tpr, _ = roc_curve(y_test, y_score)
                        fig, ax = plt.subplots()
                        ax.plot(fpr, tpr, label=f"ROC AUC = {auc(fpr, tpr):.2f}")
                        ax.plot([0, 1], [0, 1], linestyle="--")
                        ax.legend()
                        st.pyplot(fig)
                    except:
                        st.warning("ROC not available")

                    try:
                        explainer = shap.Explainer(model, X_train)
                        shap_vals = explainer(X_test[:100])
                        st.subheader("SHAP Summary")
                        fig, ax = plt.subplots()
                        shap.plots.beeswarm(shap_vals, show=False)
                        st.pyplot(fig)
                    except:
                        st.warning("SHAP not available")

                st.header("📊 AI Reasoning Report")
                report = self.prompter.generate_reasoning_report(
                    hypothesis, [f["name"] for f in selected_defs], df
                )
                st.markdown(report)
                path = self.modeler.generate_pdf_report(report)
                if path:
                    with open(path, "rb") as f:
                        st.download_button(
                            "Download Report as PDF", f, file_name="AI_Report.pdf"
                        )


if __name__ == "__main__":
    app = HypothesisAutoMLApp()
    app.run()
