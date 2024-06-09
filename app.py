
import streamlit as st #for UI and application development
import pandas as pd #for data handling and manipulation
import plotly.express as px #for interactive visualizations

#to handle file paths and env variables
import os
import tempfile
from dotenv import load_dotenv

#for chatbot - Google's Gemini
from langchain.agents.agent_types import AgentType 
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_google_genai import ChatGoogleGenerativeAI

def main(): 
    load_dotenv() #load environment variables from .env file for API keys
    api_key = os.getenv('GOOGLE_API_KEY')#fetching the api key from the .env file

    #defining UI layout and title
    st.set_page_config(page_title='CSV Analysis and Query Application', page_icon="🗃️", layout="centered")
    st.title('CSV Analysis and Query Application')

    #Initializing session state (so that we can switch between the chatbot and analysis application without having the user to upload the csv everytime)
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'tmp_file_path' not in st.session_state:
        st.session_state.tmp_file_path = None

    #sidebar
    st.sidebar.subheader("File Options")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv") #upload csv

    if uploaded_file is not None:
        #clean up the previous temporary file path if it exists
        if st.session_state.tmp_file_path and os.path.exists(st.session_state.tmp_file_path):
            os.remove(st.session_state.tmp_file_path)
            st.session_state.tmp_file_path = None

        st.session_state.uploaded_file = uploaded_file
        #to handle different csv format options
        st.sidebar.subheader('CSV Format Options')
        delimiter_options = [',', ';', '|'] #the csv files can have different delimeters
        delimiter = st.sidebar.selectbox("Select delimiter", delimiter_options, index=0)
        header = st.sidebar.checkbox("File has header", value=True) #to handle csv files with and without header

        #loading the CSV file
        try:
            df = pd.read_csv(uploaded_file, delimiter=delimiter, header=0 if header else None)
            st.sidebar.success(f"CSV file loaded successfully with delimiter '{delimiter}'", icon="✅") #success if file loaded
            st.session_state.df = df

            #saving the uploaded file to a temp location, because the csv agent, can only access the file through the file path and cannot directly access the file content
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
                tmp.write(uploaded_file.getvalue())
                st.session_state.tmp_file_path = tmp.name

        except Exception as e: #exception handling, if file not loaded due to an error, this will display the error
            st.sidebar.error(f"Error loading file with delimiter '{delimiter}': {e}")

    if st.session_state.df is not None:
        df = st.session_state.df

        #display the data, user can sort, select and interact with the dataframe
        st.write("Data from CSV file:")
        st.dataframe(df)

        #to select the chatbot or analysis application
        option = st.radio('Select an option:', ['CSV Analysis', 'Query your CSV: CSVBot'], horizontal=True)

        if option == 'CSV Analysis':
            st.header('CSV Analysis')

            #data validation required, as we can't allow user to select categorical columns for scatter plot
            numerical_columns = df.select_dtypes(include=['number']).columns
            categorical_columns = df.select_dtypes(include=['object', 'category', 'int64', 'int32']).columns
            strictly_cat_columns = df.select_dtypes(include=['object', 'category']).columns

            with st.expander('Scatter Plot'): #allowing user to select features and colour to plot and then plotting the graph using plotly
                st.subheader('Scatter Plot')
                x_axis = st.selectbox('Select X-axis', numerical_columns)
                y_axis = st.selectbox('Select Y-axis', numerical_columns)
                color = st.selectbox('Select Color', ['None'] + list(categorical_columns))

                if st.button('Generate Scatter Plot'):
                    if x_axis and y_axis:
                        st.write(f'Scatter plot of {x_axis} vs {y_axis}')
                        fig = px.scatter(df, x=x_axis, y=y_axis, color=color if color != 'None' else None)
                        st.plotly_chart(fig, use_container_width=True) #making sure the graph stays within the expander

            with st.expander('Univariate Analysis'): #user can select the type of count plot needed, such as bar and pie and also select the feature
                st.subheader('Univariate Analysis')
                plot_type = st.selectbox('Select plot type', ['Bar Chart', 'Pie Chart'])
                selected_column = st.selectbox('Select column', strictly_cat_columns) #only allowing to select from categorical columns

                if st.button('Generate Plot'):
                    if plot_type == 'Bar Chart':
                        fig = px.bar(df, x=selected_column)
                        st.write(f'Count Plot: Bar Chart of {selected_column}')
                        st.plotly_chart(fig, use_container_width=True)
                    elif plot_type == 'Pie Chart':
                        fig = px.pie(df, names=selected_column)
                        st.write(f'Count Plot: Pie Chart of {selected_column}')
                        st.plotly_chart(fig, use_container_width=True)

            #additional feature where user can select three features and visualize the scatter plot
            with st.expander('3D Scatter Plot'): 
                st.subheader('3D Scatter Plot Analysis')
                x_axis_3d = st.selectbox('Select X axis', numerical_columns)
                y_axis_3d = st.selectbox('Select Y axis', numerical_columns)
                z_axis_3d = st.selectbox('Select Z axis', numerical_columns)
                color_3d = st.selectbox('Select Hue/ Color', ['None'] + list(categorical_columns))

                if st.button('Generate 3D Scatter Plot'):
                    if x_axis_3d and y_axis_3d and z_axis_3d:
                        st.write(f'3D Scatter plot of {x_axis_3d} vs {y_axis_3d} vs {z_axis_3d}')
                        fig = px.scatter_3d(df, x=x_axis_3d, y=y_axis_3d, z=z_axis_3d, color=color_3d if color_3d != 'None' else None)
                        st.plotly_chart(fig, use_container_width=True)
        
        #will get activated if user selects the chatbot
        elif option == 'Query your CSV: CSVBot':
            st.header('Chat with CSV Agent: CSVBot')
            question = st.text_input('Enter your question about the data:')

            if st.button('Ask'):
                #prompt engineering for better articulation of the responses from the llm
                prompt = f"Based on the provided data, give a well-structured and informative answer to the following question: {question}. Ensure that the answer should be relevant or related to the data. If the data does not contain information relevant to the question, state that you don't know the answer and do not fabricate any information. Note that you are a CSVBot specially created for Newgen Software to chat with their csv files."
                
                try:
                    #creating the agent from langchain, with Gemini as the LLM and the uploaded file as input (using the temporary file path from session state)
                    agent = create_csv_agent(
                        ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key, temperature=0.8),#temperature=0.8 because want the model to give more deterministic and conservative response, and stick mainly to the csv data
                        st.session_state.tmp_file_path,
                        verbose=True,
                        #Zero-shot means the agent functions on the current action only, it does a reasoning step before acting
                        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    )
                    answer = agent.run(prompt)
                    st.write("CSVBot's Answer:")
                    st.write(answer)

                except Exception as e:#error handling, if unable to process query
                    st.error(f"Error querying the data: {e}")

    else:
        st.write("Please upload a CSV file to get started.")

if __name__ == '__main__':
    main()
