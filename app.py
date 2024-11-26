import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import os

from dotenv import load_dotenv
load_dotenv()

# Configure the API key for Google Generative AI
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize the generative model
model = genai.GenerativeModel("gemini-1.5-flash")

def generate_advanced_plot_code_gemini(query, df):
    data_structure = {col: str(dtype) for col, dtype in df.dtypes.items()}
    prompt = (
        f"Generate Python code to create a plot for this query: '{query}'.\n"
        f"The DataFrame has the following columns and data types:\n{data_structure}.\n"
        "Use matplotlib, seaborn, or plotly for the plot and ensure the code is executable with necessary imports. "
        "Just give me pure code without any explanation or anything, so if I run it on Python it runs without alteration and i am executing it with with st.echo():  # Show generated code in the app exec(code, {'df': data, 'plt': plt, 'sns': sns})"
    )
    response = model.generate_content(prompt)
    return response.text

def clean_data(df):
    df = df.dropna(thresh=len(df) * 0.5, axis=1)
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].mean())
    for col in df.select_dtypes(include=[object]).columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df

def generate_automatic_plots(df):
    plots = []

    # Numerical columns distribution
    num_cols = df.select_dtypes(include=[np.number]).columns[:2]
    if len(num_cols) > 0:
        fig = make_subplots(rows=1, cols=len(num_cols), subplot_titles=[f'Distribution of {col}' for col in num_cols])
        for i, col in enumerate(num_cols, 1):
            fig.add_trace(go.Histogram(x=df[col], name=col), row=1, col=i)
        fig.update_layout(height=400, width=800, title_text="Numerical Distributions")
        plots.append(fig)

    # Categorical columns distribution
    cat_cols = df.select_dtypes(include=[object]).columns[:2]
    if len(cat_cols) > 0:
        figs = []
        for col in cat_cols:
            value_counts = df[col].value_counts()
            fig = go.Figure(data=[go.Pie(labels=value_counts.index, values=value_counts.values, name=col)])
            fig.update_layout(height=400, width=400, title_text=f"Distribution of {col}")
            figs.append(fig)
        plots.extend(figs)

    # Scatter plot (if applicable)
    if len(num_cols) >= 2:
        fig = px.scatter(df, x=num_cols[0], y=num_cols[1], color=cat_cols[0] if len(cat_cols) > 0 else None,
                         title=f"Scatter Plot: {num_cols[0]} vs {num_cols[1]}")
        plots.append(fig)

    # Line plot (if applicable, assuming first numerical column could be a time series)
    if len(num_cols) > 0:
        fig = px.line(df, y=num_cols[0], title=f"Line Plot: {num_cols[0]} Over Index")
        plots.append(fig)

    # Interactive Correlation heatmap for numerical columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 1:
        corr_matrix = df[num_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            text=np.round(corr_matrix, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False))
        
        fig.update_layout(
            title='Interactive Correlation Heatmap',
            height=600,
            width=800,
            xaxis_title="Features",
            yaxis_title="Features"
        )
        plots.append(fig)

    return plots[:5]  # Limit to 5 plots

# Streamlit UI components
st.set_page_config(layout="wide")
st.title("📊 Smart Data Visualizer")


st.write("Upload a CSV file to automatically generate important visualizations, then query for custom plots!")

# File uploader
uploaded_file = st.file_uploader("📂 Please upload a dataset to get started.", type="csv")
if uploaded_file:
    try:
        # Detect delimiter and handle missing headers
        try:
            # Attempt to read with default delimiter
            data = pd.read_csv(uploaded_file)
        except pd.errors.ParserError:
            # Fallback to semicolon-separated data
            uploaded_file.seek(0)  # Reset file pointer
            data = pd.read_csv(uploaded_file, delimiter=';')

        # Check if column names are missing
        if data.columns.str.match(r"Unnamed:").all():
            data = pd.read_csv(uploaded_file, header=None)  # Reload without headers
            data.columns = [f"Column_{i+1}" for i in range(data.shape[1])]  # Assign default column names

        # Clean data (handle missing values)
        data = clean_data(data)
        st.success("✅ Dataset Loaded and Cleaned Successfully!")

        with st.expander("👀 View Dataset Preview"):
            st.write(data.head())

        # Generate and display automatic plots
        st.header("📈 Automated Visualizations")
        auto_plots = generate_automatic_plots(data)

        # Create a 2x3 grid for plots
        col1, col2 = st.columns(2)
        for i, fig in enumerate(auto_plots):
            if i % 2 == 0:
                with col1:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                with col2:
                    st.plotly_chart(fig, use_container_width=True)

        # Query input
        st.header("🗣 Query-Based Visualization")
        query_completed = True
        count = 1

        while query_completed:
            query_completed = False
            query = st.text_input("Enter your query for the plot", key=f"query-{count}")
            if query:
                with st.spinner("Generating custom plot..."):
                    # Generate and execute plot code
                    code = generate_advanced_plot_code_gemini(query, data)
                    code = code.strip().splitlines()
                    if len(code) > 2:
                        code = "\n".join(code[1:-1])

                    # Execute the generated code
                    with st.echo():
                        exec(code, {"df": data, "plt": plt, "sns": sns})

                    # Display plot
                    st.pyplot(plt.gcf())
                    plt.close()  # Close the figure to free up memory

                # Clear the query input for the next query
                query = ""
                count += 1
                query_completed = True

    except pd.errors.EmptyDataError:
        st.error("The uploaded file is empty. Please upload a valid CSV file.")
    except pd.errors.ParserError:
        st.error("Error parsing the CSV file. Please ensure it's a valid CSV format.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")

else:
    st.info("Upload your dataset and explore the visualizations.")
    
# Add authors
st.markdown("""
**Authors:**
- Pooja Patil : 2022301011
- Darshana Chothave : 2021300021
- Pratiksha Patil: 2022301016
""")