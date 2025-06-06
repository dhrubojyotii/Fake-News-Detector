import streamlit as st
import numpy as np
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
from pathlib import Path
import time

# Configure page settings
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 3rem;
    }
    
    .input-section {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid #e9ecef;
        margin-bottom: 2rem;
    }
    
    .result-section {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.1rem;
        font-weight: bold;
    }
    
    .fake-result {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    
    .real-result {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    .info-card {
        background-color: #e3f2fd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    
    .stats-container {
        display: flex;
        justify-content: space-around;
        margin: 2rem 0;
    }
    
    .stat-item {
        text-align: center;
        padding: 1rem;
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        min-width: 120px;
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
    }
    
    .stat-label {
        color: #666;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
    st.session_state.model = None
    st.session_state.vectorizer = None

# Download stopwords (if needed)
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('stopwords', quiet=True)
        return True
    except Exception as e:
        st.error(f"Error downloading NLTK data: {str(e)}")
        return False

# Initialize PorterStemmer
ps = PorterStemmer()

# Function for text preprocessing (stemming)
def stemming(content):
    if not content or pd.isna(content):
        return ""
    stemmed_content = re.sub('[^a-zA-Z]', ' ', str(content))
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    return ' '.join(stemmed_content)

# Load and preprocess dataset (cached for performance)
@st.cache_data
def load_and_process_data():
    try:
        data_path = Path('train.csv')
        if not data_path.exists():
            return None, "train.csv file not found! Please make sure it's in the same directory as the script."
        
        news_df = pd.read_csv(data_path).fillna(' ')
        news_df['content'] = news_df['author'] + ' ' + news_df['title']
        news_df['content'] = news_df['content'].apply(stemming)
        
        return news_df, None
    except Exception as e:
        return None, f"Error loading data: {str(e)}"

# Vectorization and model training
@st.cache_resource
def train_news_model(news_df):
    try:
        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(news_df['content']).toarray()
        y = news_df['label'].values
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)
        
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, Y_train)
        
        # Calculate accuracy
        train_accuracy = model.score(X_train, Y_train)
        test_accuracy = model.score(X_test, Y_test)
        
        return model, vectorizer, train_accuracy, test_accuracy, len(news_df)
    except Exception as e:
        return None, None, None, None, None

# Main UI
def main():
    # Header
    st.markdown('<h1 class="main-header">📰 Fake News Detector</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Detect fake news using advanced machine learning algorithms</p>', unsafe_allow_html=True)
    
    # Initialize NLTK data
    if not download_nltk_data():
        st.stop()
    
    # Load model with progress indicator
    if not st.session_state.model_loaded:
        with st.spinner('🔄 Loading and training the model... This may take a moment.'):
            progress_bar = st.progress(0)
            
            # Load data
            progress_bar.progress(25)
            news_df, error = load_and_process_data()
            
            if error:
                st.error(f"❌ {error}")
                st.info("💡 **Tip:** Make sure you have the 'train.csv' file in the same directory as this script.")
                st.stop()
            
            progress_bar.progress(50)
            
            # Train model
            model, vectorizer, train_acc, test_acc, dataset_size = train_news_model(news_df)
            progress_bar.progress(100)
            
            if model is None:
                st.error("❌ Failed to train the model. Please check your data format.")
                st.stop()
            
            # Store in session state
            st.session_state.model = model
            st.session_state.vectorizer = vectorizer
            st.session_state.train_accuracy = train_acc
            st.session_state.test_accuracy = test_acc
            st.session_state.dataset_size = dataset_size
            st.session_state.model_loaded = True
            
            progress_bar.empty()
            st.success("✅ Model loaded successfully!")
            time.sleep(1)
            st.rerun()
    
    # Model statistics
    if st.session_state.model_loaded:
        st.markdown("### 📊 Model Performance")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="stat-item">
                <div class="stat-number">{st.session_state.train_accuracy:.1%}</div>
                <div class="stat-label">Training Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="stat-item">
                <div class="stat-number">{st.session_state.test_accuracy:.1%}</div>
                <div class="stat-label">Test Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="stat-item">
                <div class="stat-number">{st.session_state.dataset_size:,}</div>
                <div class="stat-label">Training Articles</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Input section
    st.markdown("### 🔍 News Article Analysis")
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["📝 Manual Input", "📄 Example Articles"])
    
    with tab1:
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        
        input_text = st.text_area(
            "Enter the news article text below:",
            height=200,
            placeholder="Paste your news article here...",
            help="Enter the complete news article text for analysis"
        )
        
        # Analysis button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_button = st.button("🔍 Analyze Article", type="primary", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Prediction and results
        if analyze_button and input_text.strip():
            if st.session_state.model_loaded:
                with st.spinner('🔍 Analyzing the article...'):
                    processed_text = stemming(input_text)
                    
                    if not processed_text.strip():
                        st.warning("⚠️ The article doesn't contain enough meaningful text for analysis.")
                    else:
                        transformed_text = st.session_state.vectorizer.transform([processed_text]).toarray()
                        prediction = st.session_state.model.predict(transformed_text)[0]
                        confidence = st.session_state.model.predict_proba(transformed_text)[0]
                        
                        # Display results
                        st.markdown("### 📋 Analysis Results")
                        
                        if prediction == 1:
                            st.markdown(f"""
                            <div class="result-section fake-result">
                                🚨 <strong>FAKE NEWS DETECTED</strong><br>
                                Confidence: {max(confidence):.1%}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown("""
                            <div class="info-card">
                                <strong>⚠️ Warning:</strong> This article appears to contain misleading or false information. 
                                Please verify the information from reliable sources before sharing.
                            </div>
                            """, unsafe_allow_html=True)
                            
                        else:
                            st.markdown(f"""
                            <div class="result-section real-result">
                                ✅ <strong>LIKELY AUTHENTIC NEWS</strong><br>
                                Confidence: {max(confidence):.1%}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown("""
                            <div class="info-card">
                                <strong>✅ Good:</strong> This article appears to be legitimate news. 
                                However, always cross-reference important information with multiple reliable sources.
                            </div>
                            """, unsafe_allow_html=True)
            else:
                st.error("❌ Model not loaded. Please refresh the page.")
        
        elif analyze_button:
            st.warning("⚠️ Please enter some text to analyze.")
    
    with tab2:
        st.markdown("**Try these example articles:**")
        
        examples = {
            "Example Fake News": "BREAKING: Scientists discover that drinking coffee backwards can reverse aging process completely. Local man aged 80 now looks 25 after following this simple trick doctors don't want you to know.",
            "Example Real News": "The stock market showed mixed results today as investors weighed concerns about inflation against positive earnings reports from major technology companies. The Dow Jones fell 0.3% while the Nasdaq gained 0.7%."
        }
        
        for title, text in examples.items():
            if st.button(f"📄 {title}", key=title):
                st.text_area("Selected example:", value=text, height=100, disabled=True)
                
                if st.session_state.model_loaded:
                    processed_text = stemming(text)
                    transformed_text = st.session_state.vectorizer.transform([processed_text]).toarray()
                    prediction = st.session_state.model.predict(transformed_text)[0]
                    confidence = st.session_state.model.predict_proba(transformed_text)[0]
                    
                    if prediction == 1:
                        st.markdown(f"""
                        <div class="result-section fake-result">
                            🚨 <strong>FAKE NEWS DETECTED</strong><br>
                            Confidence: {max(confidence):.1%}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-section real-result">
                            ✅ <strong>LIKELY AUTHENTIC NEWS</strong><br>
                            Confidence: {max(confidence):.1%}
                        </div>
                        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🤖 <strong>Fake News Detector</strong> | Powered by Machine Learning</p>
        <p style="font-size: 0.9rem;">
            This tool uses natural language processing and logistic regression to analyze news articles.<br>
            Always verify information from multiple reliable sources.
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()