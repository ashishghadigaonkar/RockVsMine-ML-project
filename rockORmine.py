# 1. FIRST LINE MUST BE STREAMLIT IMPORT
import streamlit as st

# 2. IMMEDIATELY CALL set_page_config()
st.set_page_config(
    page_title="Sonar Rock vs Mine Classifier",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 3. NOW IMPORT ALL OTHER LIBRARIES
import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (accuracy_score, precision_score, 
                            recall_score, confusion_matrix, 
                            classification_report, roc_curve, auc,
                            f1_score, roc_auc_score, precision_recall_curve)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.dataframe_explorer import dataframe_explorer

# Load individual models and components from pickle files
@st.cache_resource
def load_models_and_data():
    try:
        # Verify all required files exist first
        required_files = ['knn_model.pkl', 'rf_model.pkl', 'svm_model.pkl', 'scaler.pkl', 'test_data.pkl']
        for file in required_files:
            if not os.path.exists(file):
                raise FileNotFoundError(f"{file} not found")
        
        # Load models with verification
        def safe_load_model(filepath):
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
                if not hasattr(model, 'predict'):  # Basic model verification
                    raise ValueError(f"Invalid model in {filepath}")
                return model
        
        knn_model = safe_load_model('knn_model.pkl')
        rf_model = safe_load_model('rf_model.pkl')
        svm_model = safe_load_model('svm_model.pkl')
        
        # Load scaler
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
            if not hasattr(scaler, 'transform'):
                raise ValueError("Invalid scaler object")
        
        # Load test data
        with open('test_data.pkl', 'rb') as f:
            test_data = pickle.load(f)
            if 'X_test' not in test_data or 'y_test' not in test_data:
                raise ValueError("Invalid test data format")
            X_test, y_test = test_data['X_test'], test_data['y_test']
        
        models = {
            'K-Nearest Neighbors': knn_model,
            'Random Forest': rf_model,
            'Support Vector Machine': svm_model
        }
        
        return models, scaler, X_test, y_test
        
    except (FileNotFoundError, ValueError, pickle.PickleError) as e:
        st.warning(f"Model loading failed: {str(e)}. Training new models...")
        
        # Load and prepare data
        data = pd.read_csv('sonar data (1).csv', header=None)
        X = data.drop(60, axis=1)
        y = data[60].map({'R': 0, 'M': 1})
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create and fit scaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Create and fit models
        knn_model = KNeighborsClassifier(n_neighbors=5).fit(X_train_scaled, y_train)
        rf_model = RandomForestClassifier(n_estimators=100).fit(X_train_scaled, y_train)
        svm_model = SVC(probability=True).fit(X_train_scaled, y_train)
        
        # Save to individual pickle files
        def safe_save(obj, filepath):
            with open(filepath, 'wb') as f:
                pickle.dump(obj, f)
        
        safe_save(knn_model, 'knn_model.pkl')
        safe_save(rf_model, 'rf_model.pkl')
        safe_save(svm_model, 'svm_model.pkl')
        safe_save(scaler, 'scaler.pkl')
        safe_save({'X_test': X_test_scaled, 'y_test': y_test}, 'test_data.pkl')
        
        models = {
            'K-Nearest Neighbors': knn_model,
            'Random Forest': rf_model,
            'Support Vector Machine': svm_model
        }
        
        return models, scaler, X_test_scaled, y_test
models, scaler, X_test, y_test = load_models_and_data()

# Calculate model metrics
@st.cache_data
def calculate_model_metrics():
    metrics = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        metrics[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba) if y_proba is not None else None
        }
    
    # Find best model based on accuracy
    best_model = max(models.keys(), key=lambda x: metrics[x]['accuracy'])
    best_accuracy = metrics[best_model]['accuracy']
    
    return metrics, best_model, best_accuracy

model_metrics, best_model, best_accuracy = calculate_model_metrics()

# Load full dataset separately
@st.cache_data
def load_full_data():
    return pd.read_csv('sonar data (1).csv', header=None)

full_data = load_full_data()

# Custom CSS for enhanced UI
@st.cache_data
def get_css():
    return """
    <style>
        /* Main background */
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #2c3e50 0%, #1a2530 100%) !important;
            color: white !important;
        }
        
        /* Button styling */
        .stButton>button {
            background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 24px;
            font-weight: 600;
            transition: all 0.3s;
        }
        
        .stButton>button:hover {
            transform: scale(1.05);
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }
        
        /* Slider styling */
        .stSlider .thumb {
            background-color: #4b6cb7 !important;
        }
        
        /* Metric cards */
        .metric-card {
            background: white;
            border-radius: 12px;
            padding: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: all 0.3s;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.15);
        }
        
        /* Danger animations */
        .danger-animation {
            animation: danger-pulse 1.5s infinite;
        }
        @keyframes danger-pulse {
            0% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.1); opacity: 0.8; }
            100% { transform: scale(1); opacity: 1; }
        }
        .mine-icon {
            font-size: 5rem;
            text-align: center;
            margin: 20px 0;
        }
        @keyframes explosion {
            0% { transform: scale(1); opacity: 1; }
            50% { transform: scale(3); opacity: 0.8; }
            100% { transform: scale(1); opacity: 1; }
        }
        .explosion {
            animation: explosion 1s;
            text-align: center;
            font-size: 5rem;
        }
    </style>
    """

st.markdown(get_css(), unsafe_allow_html=True)

# Sidebar for navigation
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2069/2069753.png", width=80)
    st.title("Sonar Classifier")
    st.markdown("""
    <div style="margin-top:-20px; margin-bottom:20px; color:white; font-size:14px;">
    Classify sonar signals as rocks or mines
    </div>
    """, unsafe_allow_html=True)
    
    page = st.radio("Navigation", ["🏠 Dashboard", "🔮 Live Prediction", "📊 Model Analysis", "⚙️ Settings"])
    
    st.markdown("---")
    st.markdown("""
    <div style="color:#aaa; font-size:12px;">
    <b>Model Information</b><br>
    • KNN (k=5)<br>
    • Random Forest (100 trees)<br>
    • SVM (RBF kernel)
    </div>
    """, unsafe_allow_html=True)

# Dashboard Page
if page == "🏠 Dashboard":
    st.title("Sonar Rock vs Mine Classification")
    st.markdown("""
    <div style="background:linear-gradient(90deg, #4b6cb7 0%, #182848 100%); 
                padding:20px; border-radius:12px; color:white;">
    <h3 style="color:white; margin-top:0;">Real-time sonar signal classification system</h3>
    <p>This application uses machine learning to distinguish between rocks and mines using sonar frequency data.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics row
    col1, col2, col3 = st.columns(3)
    with col1:
        with stylable_container(
            key="metric1",
            css_styles="""
            {
                background: white;
                border-radius: 12px;
                padding: 15px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            """
        ):
            st.metric("Models Available", len(models))
    with col2:
        with stylable_container(
            key="metric2",
            css_styles="""
            {
                background: white;
                border-radius: 12px;
                padding: 15px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            """
        ):
            st.metric("Features Analyzed", X_test.shape[1])
    with col3:
        with stylable_container(
            key="metric3",
            css_styles="""
            {
                background: white;
                border-radius: 12px;
                padding: 15px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            """
        ):
            st.metric("Best Accuracy", f"{best_accuracy:.1%}", best_model)
    style_metric_cards()
    
    # Interactive data explorer
    st.subheader("Dataset Explorer")
    filtered_df = dataframe_explorer(full_data)
    st.dataframe(filtered_df, use_container_width=True)
    
    # Feature distribution visualization
    st.subheader("Feature Distribution")
    selected_feature = st.selectbox("Select feature to visualize", options=range(60), format_func=lambda x: f"Feature {x+1}")
    
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(full_data[selected_feature], kde=True, ax=ax)
    ax.set_title(f"Distribution of Feature {selected_feature+1}")
    st.pyplot(fig)

# Live Prediction Page
elif page == "🔮 Live Prediction":
    st.title("Real-time Sonar Classification")
    
    # Initialize features
    if 'features' not in st.session_state:
        st.session_state.features = [0.5] * 60
    
    # Model selection with dynamic accuracy indicators
    st.subheader("Select Classification Model")
    model_option = st.radio(
        "Choose your model:",
        options=list(models.keys()),
        horizontal=True,
        label_visibility="collapsed"
    )
    
    # Dynamic model confidence indicators
    accuracy = model_metrics[model_option]['accuracy']
    if accuracy > 0.9:
        st.success(f"✅ High Reliability Model ({accuracy:.1%} accuracy)")
    elif accuracy > 0.8:
        st.warning(f"⚠️ Medium Reliability Model ({accuracy:.1%} accuracy)")
    else:
        st.error(f"❗ Lower Reliability Model ({accuracy:.1%} accuracy)")

    # Rest of the Live Prediction page remains the same...
    # [Previous implementation of input methods and prediction logic]
    # Input methods with danger-themed warnings
    st.subheader("Input Sonar Frequency Data")
    input_mode = st.radio("Input method:", 
                        ["Manual Entry", "Random Sample", "Upload CSV", "Danger Zone Examples"],
                        horizontal=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if input_mode == "Manual Entry":
            with st.expander("⚠️ Adjust Frequency Features - Caution Required", expanded=True):
                features = []
                for i in range(60):
                    features.append(st.slider(
                        f"🚨 F{i+1}",
                        min_value=0.0,
                        max_value=1.0,
                        value=st.session_state.features[i],
                        step=0.01,
                        key=f"feature_{i}"
                    ))
                st.session_state.features = features
        
        elif input_mode == "Random Sample":
            sample_type = st.selectbox("Sample danger level:", ["Safe (Rock)", "Danger (Mine)"])
            if st.button("💣 Generate Sample"):
                if sample_type == "Safe (Rock)":
                    features = [np.random.normal(0.3, 0.1).clip(0,1) for _ in range(60)]
                    st.session_state.last_sample_type = "rock"
                else:
                    features = [np.random.normal(0.7, 0.1).clip(0,1) for _ in range(60)]
                    st.session_state.last_sample_type = "mine"
                st.session_state.features = features
                st.toast(f"{'DANGER' if sample_type == 'Danger (Mine)' else 'SAFE'} sample generated!", icon="⚠️" if sample_type == 'Danger (Mine)' else "✅")
        
        elif input_mode == "Upload CSV":
            uploaded_file = st.file_uploader("🚨 Upload Danger Data", type=["csv"])
            if uploaded_file:
                try:
                    uploaded_data = pd.read_csv(uploaded_file, header=None)
                    if uploaded_data.shape[1] == 60:
                        st.session_state.features = uploaded_data.iloc[0].values.tolist()
                        st.toast("Data loaded - extreme caution advised!", icon="⚠️")
                    else:
                        st.error("❌ Invalid format - 60 features required")
                except Exception as e:
                    st.error(f"💥 Error: {str(e)}")
        
        elif input_mode == "Danger Zone Examples":
            example = st.selectbox("Select threat scenario:", 
                                 ["Clear Waters (Rock)", 
                                  "Confirmed Mine", 
                                  "Ambiguous Signal"])
            if example == "Clear Waters (Rock)":
                features = [0.2 + 0.6*(i/60) + np.random.normal(0, 0.03) for i in range(60)]
            elif example == "Confirmed Mine":
                features = [0.7 - 0.4*(i/60) + np.random.normal(0, 0.03) for i in range(60)]
            else:
                features = [0.5 + 0.2*np.sin(i/5) for i in range(60)]
            st.session_state.features = features
            st.toast(f"Loaded {example} scenario", icon="💣" if "Mine" in example else "✅")

    with col2:
        if len(st.session_state.features) == 60:
            # Threat visualization
            st.subheader("Threat Assessment")
            
            # Immediate danger indicator
            avg_value = np.mean(st.session_state.features)
            danger_meter = st.progress(0)
            if avg_value > 0.6:
                danger_meter.progress(int(avg_value*100), "HIGH DANGER")
                st.markdown("<div class='mine-icon'>💣💣💣</div>", unsafe_allow_html=True)
            elif avg_value > 0.45:
                danger_meter.progress(int(avg_value*100), "POSSIBLE THREAT")
                st.markdown("<div class='mine-icon'>💣</div>", unsafe_allow_html=True)
            else:
                danger_meter.progress(int(avg_value*100), "LOW DANGER")
                st.markdown("<div class='mine-icon'>✅</div>", unsafe_allow_html=True)
            
            # Prediction button with danger styling
            if st.button("🚨 ANALYZE FOR EXPLOSIVE THREATS", 
                        type="primary", 
                        help="Extreme caution - verify results before proceeding"):
                try:
                    input_data = np.array(st.session_state.features).reshape(1, -1)
                    scaled_data = scaler.transform(input_data)
                    model = models[model_option]
                    prediction = model.predict(scaled_data)
                    proba = model.predict_proba(scaled_data)
                    
                    # Danger-themed results
                    if prediction[0] == 1:  # Mine
                        st.markdown("""
                        <div class="explosion">💥</div>
                        """, unsafe_allow_html=True)
                        
                        # Danger alert
                        with stylable_container(
                            key="danger_alert",
                            css_styles="""
                            {
                                background-color: #ff4444;
                                color: white;
                                border-radius: 12px;
                                padding: 20px;
                                margin: 20px 0;
                                border-left: 6px solid #ff0000;
                                animation: danger-pulse 2s infinite;
                            }
                            """
                        ):
                            st.markdown("""
                            <div style="text-align:center;">
                            <h2>⚠️ EXTREME DANGER ⚠️</h2>
                            <p>EXPLOSIVE THREAT DETECTED</p>
                            <p>Evacuate area immediately!</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Safety protocol
                        with st.expander("🚨 EMERGENCY PROTOCOL", expanded=True):
                            st.error("""
                            **IMMEDIATE ACTION REQUIRED:**
                            1. Clear the area within 500m radius
                            2. Activate emergency beacon
                            3. Notify explosive ordnance disposal team
                            4. Maintain safe distance until all-clear given
                            """)
                        
                    else:  # Rock
                        st.markdown("""
                        <div style="text-align:center; font-size:5rem; margin:20px 0;">
                        ✅
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with stylable_container(
                            key="safe_alert",
                            css_styles="""
                            {
                                background-color: #00C851;
                                color: white;
                                border-radius: 12px;
                                padding: 20px;
                                margin: 20px 0;
                                border-left: 6px solid #007E33;
                            }
                            """
                        ):
                            st.markdown("""
                            <div style="text-align:center;">
                            <h2>✅ AREA SECURE</h2>
                            <p>No explosive threats detected</p>
                            <p>Proceed with caution</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Threat assessment details
                    with st.expander("🔍 THREAT ASSESSMENT DETAILS"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Explosive Probability", 
                                     f"{proba[0][1]*100:.1f}%", 
                                     delta="+DANGER" if proba[0][1] > 0.5 else "-SAFE",
                                     delta_color="inverse")
                        with col2:
                            st.metric("Safety Confidence", 
                                     f"{max(proba[0])*100:.1f}%",
                                     help="Confidence in this assessment")
                        
                        # Feature danger hotspots
                        st.write("**Most dangerous frequencies:**")
                        top_danger = np.argsort(st.session_state.features)[-3:][::-1]
                        for i, idx in enumerate(top_danger):
                            danger_pct = int(st.session_state.features[idx]*100)
                            st.progress(danger_pct, f"Frequency {idx+1}: {danger_pct}% danger indicator")
                
                except Exception as e:
                    st.error(f"💣 SYSTEM FAILURE: {str(e)}")
                    st.markdown("""
                    <div style="text-align:center; font-size:3rem;">
                    🚧
                    </div>
                    <p style="text-align:center;">Classification system offline</p>
                    <p style="text-align:center;">Contact explosive ordnance team immediately</p>
                    """, unsafe_allow_html=True)

            # Danger visualization tabs
            tab1, tab2 = st.tabs(["Threat Pattern", "Danger Map"])
            with tab1:
                fig = px.line(
                    x=range(60),
                    y=st.session_state.features,
                    title="Danger Signal Pattern",
                    labels={"x": "Frequency Band", "y": "Threat Indicator"},
                    height=400
                )
                fig.add_hrect(y0=0.6, y1=1, fillcolor="red", opacity=0.2, 
                             annotation_text="Danger Zone", annotation_position="top left")
                fig.add_hrect(y0=0.4, y1=0.6, fillcolor="orange", opacity=0.2, 
                             annotation_text="Caution Zone", annotation_position="top left")
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                # Create a danger heatmap
                danger_matrix = np.array(st.session_state.features).reshape(10, 6)
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(danger_matrix, 
                            annot=True, 
                            fmt=".2f", 
                            cmap="RdYlGn_r",
                            vmin=0, 
                            vmax=1,
                            ax=ax)
                ax.set_title("Danger Concentration Map")
                st.pyplot(fig)

# Model Analysis Page
elif page == "📊 Model Analysis":
    st.title("Model Performance Analysis")
    
    # Convert metrics to DataFrame for display
    metrics_df = pd.DataFrame.from_dict(model_metrics, orient='index')
    metrics_df = metrics_df.reset_index().rename(columns={'index': 'Model'})
    
    # Interactive metrics explorer
    st.dataframe(
        metrics_df.style.format({
            "accuracy": "{:.2%}",
            "precision": "{:.2%}",
            "recall": "{:.2%}",
            "f1": "{:.2%}",
            "roc_auc": "{:.3f}"
        }).highlight_max(color='lightgreen').highlight_min(color='#ffcccb'),
        use_container_width=True
    )
    
    # Visualization tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Metrics", "📊 Confusion Matrix", "📉 ROC Curve", "📌 Precision-Recall"])
    
    with tab1:
        selected_metric = st.selectbox("Select metric to visualize", ["accuracy", "precision", "recall", "f1"])
        fig = px.bar(metrics_df, x="Model", y=selected_metric, 
                     title=f"{selected_metric.capitalize()} Comparison", color="Model")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        selected_model = st.selectbox("Select model", list(models.keys()))
        model = models[selected_model]
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        fig = px.imshow(cm,
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=['Rock', 'Mine'],
                        y=['Rock', 'Mine'],
                        text_auto=True,
                        aspect="auto")
        fig.update_layout(title=f"{selected_model} Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        fig = go.Figure()
        fig.add_shape(type='line', line=dict(dash='dash'),
                     x0=0, x1=1, y0=0, y1=1)
        
        for name, model in models.items():
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                auc_score = auc(fpr, tpr)
                
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    name=f'{name} (AUC = {auc_score:.2f})',
                    mode='lines'
                ))
        
        fig.update_layout(
            title='ROC Curves',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain='domain'),
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        fig = go.Figure()
        
        for name, model in models.items():
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:, 1]
                precision, recall, _ = precision_recall_curve(y_test, y_proba)
                
                fig.add_trace(go.Scatter(
                    x=recall, y=precision,
                    name=f'{name}',
                    mode='lines'
                ))
        
        fig.update_layout(
            title='Precision-Recall Curves',
            xaxis_title='Recall',
            yaxis_title='Precision',
        )
        st.plotly_chart(fig, use_container_width=True)

# Settings Page
elif page == "⚙️ Settings":
    st.title("Application Settings")
    
    with st.expander("Display Options"):
        col1, col2 = st.columns(2)
        with col1:
            theme = st.selectbox("Color Theme", ["Light", "Dark", "System"])
        with col2:
            animation = st.toggle("Enable Animations", True)
    
    with st.expander("Model Configuration"):
        st.warning("Advanced settings - modify with caution")
        model_to_configure = st.selectbox("Select model to configure", list(models.keys()))
        
        if model_to_configure == "K-Nearest Neighbors":
            new_k = st.slider("Number of neighbors (k)", 1, 20, 5)
            if st.button("Update KNN Configuration"):
                models['K-Nearest Neighbors'].set_params(n_neighbors=new_k)
                # Save updated model
                with open('knn_model.pkl', 'wb') as f:
                    pickle.dump(models['K-Nearest Neighbors'], f)
                st.success("KNN configuration updated and saved!")
                # Clear cache to recalculate metrics
                st.cache_data.clear()
        
        elif model_to_configure == "Random Forest":
            new_estimators = st.slider("Number of trees", 10, 500, 100)
            if st.button("Update Random Forest Configuration"):
                models['Random Forest'].set_params(n_estimators=new_estimators)
                # Save updated model
                with open('rf_model.pkl', 'wb') as f:
                    pickle.dump(models['Random Forest'], f)
                st.success("Random Forest configuration updated and saved!")
                # Clear cache to recalculate metrics
                st.cache_data.clear()
    
    with st.expander("Data Management"):
        if st.button("Clear Cache"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Cache cleared successfully!")
    
    st.markdown("---")
    st.markdown("""
    <div style="color:#aaa; font-size:12px;">
    <b>Application Version:</b> 1.0.0<br>
    <b>Last Updated:</b> April 2024
    </div>
    """, unsafe_allow_html=True)