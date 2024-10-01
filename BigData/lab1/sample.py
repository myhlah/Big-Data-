import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Title of the app
st.title('Exploratory Data Analysis with Streamlit')

# File uploader
uploaded_file = st.file_uploader("Upload CSV file here", type="csv")

if uploaded_file is not None:
    # Load the data
    df = pd.read_csv(uploaded_file, delimiter=";")

    # Display the first few rows of the dataframe
    st.subheader('Data Preview')
    st.write(df.head())

    # Display summary statistics
    st.subheader('Summary Statistics')
    st.write(df.describe())
    
    # Create two columns for Data Info and Missing Values
    col1, col2 = st.columns(2)

    # Data Info in the first column
    with col1:
        st.subheader('Data Info')
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        filtered_info = "\n".join(s.split('\n')[1:])
        st.text(filtered_info)

    # Missing Values in the second column
    with col2:
        st.subheader('Missing Values')
        st.write(df.isnull().sum())
        df = df.fillna(df.select_dtypes(include=[float, int]).mean())

    # Create two columns for Pie Chart and Heatmap
    col1, col2 = st.columns(2)

    # Pie Chart in the first column
    with col1:
        st.subheader('Pie Chart ')
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            selected_col = st.selectbox('Select a categorical column for pie chart:', categorical_cols)

            if selected_col:
                category_counts = df[selected_col].value_counts()
                fig, ax = plt.subplots(figsize=(3,3))  
                ax.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 4},
                       wedgeprops=dict(width=0.5)) 
                st.pyplot(fig)
        else:
            st.write("No categorical columns found in the dataset.")
    
    # Heatmap in the second column
    with col2:
        st.subheader('Correlation Heatmap')
        corr = df.select_dtypes(include=[float, int]).corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax,
                    annot_kws={"size": 8}, cbar_kws={'shrink': .8}, 
                    linewidths=0.5, linecolor='gray')

        # Rotate labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)

        st.pyplot(fig)

    # Create two columns for Scatter Plot and Bar Chart
    col1, col2 = st.columns(2)

    # Scatter Plot in the first column
    with col1:
        st.subheader('Scatter Plot')
        num_cols = df.select_dtypes(include=[np.number]).columns
        x_col = st.selectbox('Select X-axis column', num_cols)
        y_col = st.selectbox('Select Y-axis column', num_cols)

        fig, ax = plt.subplots(figsize=(8, 6)) 
        sns.scatterplot(x=df[x_col], y=df[y_col], ax=ax)
        ax.set_title(f'Scatter Plot of {x_col} vs {y_col}')
        st.pyplot(fig)

    # Bar Chart in the second column
    with col2:
        st.subheader('Bar Chart')
        if len(categorical_cols) > 0:
            selected_col = st.selectbox('Select a categorical column for bar chart or "Show All":', ['Show All'] + list(categorical_cols))
            
            if selected_col:
                if selected_col == 'Show All':
                    combined_counts = pd.DataFrame()
                    for col in categorical_cols:
                        counts = df[col].value_counts().reset_index()
                        counts.columns = ['Category', 'Count']
                        counts['Source Column'] = col
                        combined_counts = pd.concat([combined_counts, counts])
                    
                    fig, ax = plt.subplots(figsize=(12, 8))  
                    sns.barplot(x='Category', y='Count', hue='Source Column', data=combined_counts, ax=ax)
                    ax.set_xlabel('Category')
                    ax.set_ylabel('Count')
                    ax.set_title('Combined Bar Chart of All Categorical Columns')
                    st.pyplot(fig)
                else:
                    category_counts = df[selected_col].value_counts()
                    fig, ax = plt.subplots(figsize=(10, 6))  
                    category_counts.plot(kind='bar', ax=ax)
                    ax.set_xlabel('Category')
                    ax.set_ylabel('Count')
                    ax.set_title(f'Bar Chart of {selected_col}')
                    st.pyplot(fig)
        else:
            st.write("No categorical columns found in the dataset.")
    
    # Create three columns for Histograms, Density Plots, and Box and Whisker Plots
    col1, col2, col3 = st.columns(3)
     
    # Plot histograms in the first column
    with col1:
        st.subheader('Histograms')
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            fig, ax = plt.subplots(figsize=(6, 4))  
            df[col].hist(ax=ax, bins=20)
            ax.set_title(f'Histogram of {col}')
            st.pyplot(fig)

    # Plot density plots in the second column
    with col2:
        st.subheader('Density Plots')
        for col in num_cols:
            fig, ax = plt.subplots(figsize=(6, 4))  
            sns.kdeplot(df[col], ax=ax, fill=True)
            ax.set_title(f'Density Plot of {col}')
            st.pyplot(fig)

    # Plot box and whisker plots in the third column
    with col3:
        st.subheader('Box and Whisker Plots')
        for col in num_cols:
            fig, ax = plt.subplots(figsize=(6, 4)) 
            sns.boxplot(x=df[col], ax=ax)
            ax.set_title(f'Box and Whisker Plot of {col}')
            st.pyplot(fig)
            #macala normailah itd105 it4d
