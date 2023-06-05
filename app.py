import os
import streamlit as st
import pandas as pd
from tabulate import tabulate
from PIL import Image
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import time
# Import the ExcelHandler class from your Jupyter Notebook
from single_utils import ExcelHandler, BayesianOptimization, GPModel
import datetime
import scipy
from scipy.stats import norm

# Set page title and favicon
st.set_page_config(
    page_title="SAGE",
    page_icon=":robot_face:",
    layout="wide",
)

col1,empty1,col2 = st.columns([4,8,4])  # creating empty slots

# Display the logo in the last column
# Load logo
logo_image = Image.open("utils/logo_small.png")
sage_logo = Image.open("utils/sagelogo.png")
col2.image(logo_image,width=180)
col1.image(sage_logo,width=150)
# Display logo and prompt user to upload Excel file
#st.image(logo_image, width=300)
#st.title("Bayesian Optimization for Smart Experimentation (BOSE)")
#st.subheader('BOSE is an application for performing active experimental design based on Bayesian Optimization.')

tab_about, tab_data, tab_opt, tab_preds = st.tabs(["ℹ️ About", "🗃 Data  ", "  📈 Optimization ", " 🔍 Visualizations"])

with tab_about:
    st.markdown(
        """
        ## 💡 What is SAGE?
        Our app, "Smart Adaptive Guidance for Experiments" (SAGE), is a suite for active
        design of experiments. By harnessing the power of the latest advancements in machine learning,
        it solves optimization problems that are complex and expensive to evaluate. Designed to be
        user-friendly and intuitive, it ensures that users, regardless of their technical background,
        can easily use the application to its full potential.
        This is the **single-objective** module, focused on optimizing a single performance metric. For more capabilities and for
        adapting the software to your specific needs please contact us.

        ## 🖥 How it works?
        The app consists of three main tabs, each with its unique functionality:

        - **🗃 Data**: This is where you upload your initial experimental data. The data should be in an Excel format. Please make
        sure you follow the "template.xslm" file provided and include both the experimental parameters (features) and the observed results (targets).
        - **📈 Optimization**: In this tab, you'll see the optimization process in action. The system will suggest the next set of experiment parameters to try based on the current available data. It uses a Bayesian Optimization approach to intelligently suggest the next experiments.
        The tool provides the flexibility to insert any query that you perform.
        - **🔍 Visualizations**: Here, you can input a set of parameters and the app will give you an estimate of the expected result, along with a probability distribution around that estimate. It allows you to explore the potential outcomes of an experiment
        without having to actually perform it. It also provides further information about the algorithm in terms of sampling utility per point.

        ## 🚀 Development
        We aim to continuously improve our application by incorporating user feedback, adding new features,
        and adapting to the ever-changing landscape of machine learning and optimization techniques. Please
        contact us for inquiries tailored to your problems.
        """
        )


with tab_data:
    uploaded_file = st.file_uploader("Please upload your Excel file:", type=['xlsm'])
    if uploaded_file is not None:
        file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
        # Initialize ExcelHandler object
        excel_handler = ExcelHandler(uploaded_file)
        # Load and display Description sheet
        description_df = excel_handler._read_variable_info()
        st.write("Problem Summary:")
        st.dataframe(description_df)

        # Check if "Observations" sheet is present
        xl = pd.ExcelFile(uploaded_file)
        if "Observations" not in xl.sheet_names:
            st.warning("The 'Observations' sheet does not seem to be present in the uploaded file. This sheet is essential for the Bayesian Optimization process. Please check your file and upload again.")
        else:
            st.success("Data from Excel file has been successfully read")
            bounds_tensor = excel_handler.get_bounds()
            # Get column names from 'Variable Name' values in feature_vars and constraint_vars dataframes
            col_names_x = excel_handler.feature_vars['Variable Name'].tolist()
            # Get the name of y variable
            target_var = excel_handler.variable_info[excel_handler.variable_info['Settings'] == 'Target']['Variable Name'].values[0]
            col_names_y = [target_var]

        if st.button('Export data'):
            if "df_train" in st.session_state:
                # Filter out the first Ninit rows
                df_export = st.session_state.df_train.iloc[st.session_state.Ninit:]
                # Save to .csv
                timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")  # current date and time
                filename = f'study_{timestamp}.csv'
                df_export.to_csv(filename, index=False)
                st.success(f"Data exported to {filename} successfully!")
            else:
                st.warning('No data to export.')




with tab_opt:
    if uploaded_file is not None:
        if "train_x" not in st.session_state or "train_y" not in st.session_state:
            #Need to initialize from excel only when app is loaded
            train_x, train_y, cons = excel_handler.observations_to_tensors()
            st.session_state.Ninit = train_x.shape[0]  # number of initial observations
            # Convert tensors to pandas dataframes
            df_train_x = pd.DataFrame(train_x.numpy(), columns=col_names_x)
            df_train_y = pd.DataFrame(train_y.numpy(), columns=col_names_y)
            # Concatenate train_x and train_y dataframes
            df_train = pd.concat([df_train_x, df_train_y], axis=1)
            # Save the dataframe to the session state
            st.session_state.df_train = df_train
            # Initialize the BO object
            st.session_state.BO = BayesianOptimization(train_X=train_x, train_Y=train_y, bounds=bounds_tensor, noiseless_obs=False)
        else:
            train_x = st.session_state.train_x
            train_y = st.session_state.train_y


        opt_col1, opt_col2 = st.columns([0.9,1.5])

        st.markdown("""
            <style>
            .stButton>button {
                background-color: lightgray;
            }
            </style>
            """, unsafe_allow_html=True)

        with opt_col1:

            # Display the dataframe
            st.markdown("##### Experimental Data")
            st.write("Add completed queries here")
            st.markdown('###')

            # Check if the "Index" column already exists
            if "Index" not in st.session_state.df_train.columns:
                # Generate the index
                index_values = list(range(-st.session_state.Ninit+1, 1)) + list(range(1, len(st.session_state.df_train) - st.session_state.Ninit + 1))
                # Insert the index as the first column
                st.session_state.df_train.insert(0, "Index", index_values)

            # Run the data editor
            st.session_state.df_train = st.experimental_data_editor(st.session_state.df_train, width=450, height=250, num_rows="dynamic")
            # Save any changes back to the session state
            st.session_state.df_train = st.session_state.df_train
            # Find rows without missing values
            complete_rows_mask = st.session_state.df_train.notna().all(axis=1)

            # Separate the target variable (y) from the features (x)
            train_x = st.session_state.df_train[complete_rows_mask].iloc[:, 1:-1]  # ignoring "Index" and assuming y is the last column
            train_y = st.session_state.df_train[complete_rows_mask].iloc[:, -1]

            # Convert pandas DataFrames to PyTorch tensors
            train_x = torch.tensor(train_x.values, dtype=torch.float32)
            train_y = torch.tensor(train_y.values, dtype=torch.float32)

            #st.markdown("""
            #<style>
            #div[data-baseweb="slider"] div:first-child div:first-child {
        #        height: 20px !important;
        #    }
            #div[data-baseweb="slider"] div:first-child div:first-child::before {
        #        height: 20px !important;
    #        }
    #        </style>
    #        """, unsafe_allow_html=True)
            st.markdown('##### Exploration Level')
            st.write("Determine the degree of exploration applied by the optimizer")
            expl_select = st.select_slider('Beta',
            options=['0.001','0.1', '0.2','0.3','0.4','0.5','0.6', '0.7', '0.8', '0.9', '1.0'],
            value='0.5',
            format_func=lambda x: 'Chill' if x == '0.001' else 'Aggressive' if x == '1.0' else x
            )
            exploration = float(expl_select)  # convert the selected value back to float for further processing

            beta = 3.0 #Baseline value for beta
            if st.button("Get Experiment"):
                st.markdown("##### Suggested Query")
                st.write("Queue of proposed experiment(s)")
                # Update the BO object

                st.session_state.BO = BayesianOptimization(train_X=train_x, train_Y=train_y, bounds=bounds_tensor, noiseless_obs=False)
                suggested_point, acq_value = st.session_state.BO.optimize_acquisition(beta=exploration*beta)
                # Convert suggested point to numpy array
                suggested_point_np = suggested_point.detach().numpy().flatten()

                # Construct a DataFrame from suggested point with appropriate column names
                st.session_state.suggested_df = pd.DataFrame([suggested_point_np], columns=col_names_x)

                # Add "Value" as column header
                st.session_state.suggested_df = st.session_state.suggested_df.rename_axis("Value", axis=1)

            # Display the stored suggested_df if it exists
            if hasattr(st.session_state, 'suggested_df'):
                table = tabulate(st.session_state.suggested_df, tablefmt="pipe", headers="keys", showindex=False)
                st.write(table, unsafe_allow_html=True)


        with opt_col2:
            #st.write(train_x)
            #st.write(train_y)
            st.markdown("##### Performance Progress")
            st.write("A visualization of the algorithm progression")
            # Get the dataframe from the session state
            df_train = st.session_state.df_train
            # Separate the target variable (y) from the features (x)
            df_train_y = df_train.iloc[:, -1]  # assuming y is the last column
            # Create a list for the x-axis
            x_axis = list(range(-st.session_state.Ninit+1, 1)) + list(range(1, df_train.shape[0]-st.session_state.Ninit+1))
            # Create a new figure
            fig = go.Figure()
            # Initialize BO in the session state if it doesn't exist
            if 'BO' not in st.session_state:
                st.session_state.BO = BayesianOptimization(train_X=train_x[:st.session_state.Ninit], train_Y=train_y[:st.session_state.Ninit], bounds=bounds_tensor, noiseless_obs=False)

            mean_init, std_dev_init = st.session_state.BO.get_posterior_stats(train_x[:st.session_state.Ninit])
            # Add trace for predicted mean with error bars for initial points
            fig.add_trace(go.Scatter(
                x=x_axis[:st.session_state.Ninit],
                y=mean_init.detach().numpy().flatten(),
                error_y=dict(
                    type='data',
                    array=1.96*std_dev_init.detach().numpy().flatten(),
                    visible=True
                ),
                mode='markers',
                marker=dict(color='lightblue', size=20),
                name='Initial Experiments',
                legendgroup="group1",
                hovertemplate="Initial Predicted Performance"
            ))

            for i in range(st.session_state.Ninit, len(df_train_y)):
                # Check if the actual observation is available
                if not np.isnan(df_train_y[i]):
                    # Update the BO object in the session state
                    if 'BO' not in st.session_state:
                        st.session_state.BO = BayesianOptimization(train_X=train_x[:i], train_Y=train_y[:i], bounds=bounds_tensor, noiseless_obs=False)
                        mean, std_dev = st.session_state.BO.get_posterior_stats(train_x[:i+1])
                    else:
                        st.session_state.BO = BayesianOptimization(train_X=train_x[:i], train_Y=train_y[:i], bounds=bounds_tensor, noiseless_obs=False)
                        mean, std_dev = st.session_state.BO.get_posterior_stats(train_x[:i+1])

                    # Add trace for predicted mean with error bars
                    fig.add_trace(go.Scatter(
                        x=x_axis[i:i+1],
                        y=mean.detach().numpy().flatten()[-1:],
                        error_y=dict(
                            type='data',
                            array=1.96*std_dev.detach().numpy().flatten()[-1:],
                            visible=True
                        ),
                        mode='markers',
                        marker=dict(color='mediumslateblue', size=20),
                        name='Predicted Performance',
                        legendgroup="group2",
                        hovertemplate="Predicted Performance at Iteration %d" % (i-st.session_state.Ninit+1),
                        showlegend= i == st.session_state.Ninit
                    ))

                    # Add trace for actual observation
                    fig.add_trace(go.Scatter(
                        x=x_axis[i:i+1],
                        y=df_train_y[i:i+1],
                        mode='markers',
                        marker=dict(color='mediumvioletred', symbol='star', size=20),
                        name='Observed Performance',
                        legendgroup="group3",
                        hovertemplate="Actual Performance at Iteration %d" % (i-st.session_state.Ninit+1),
                        showlegend= i == st.session_state.Ninit
                    ))
            # Positioning the legend and adding axis names
            fig.update_layout(
                    width=800,  # Adjust width
                    height=500,  # Adjust height
                    xaxis=dict(
                    title="Iterations",
                    titlefont=dict(
                        size=16),
                    tickfont=dict(
                        size=16),
                    showgrid = True
                ),
                yaxis=dict(
                    title="Performance",
                    titlefont=dict(
                        size=16
                    ),
                    tickfont=dict(
                        size=16
                    ),
                    showgrid=True
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="left",
                    x=0,
                    font=dict(size=15),
                ),

                shapes=[
                    dict(
                        type="line",
                        yref="paper", y0=0, y1=1,
                        xref="x", x0=0, x1=0,
                        line=dict(
                            color="Black",
                            width=1,
                            dash="dot",
                        )
                    )
                ]
            )

            # Show the plot
            st.plotly_chart(fig)
    else:
         st.write("No file has been uploaded yet.")


# Define color list
color_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

with tab_preds:
    if uploaded_file is not None:
        if "df_test" not in st.session_state:
            max_index = train_y.argmax().item()
            max_input = train_x[max_index]
            initial_test_df = pd.DataFrame(max_input.numpy()[np.newaxis, :], columns=col_names_x)
            st.session_state.df_test = initial_test_df.reset_index().rename(columns={'index': 'Index'})

        preds_col1, preds_col2 = st.columns([0.7,1.0])

        with preds_col1:
            st.markdown("##### Inspection at Desired Inputs")
            st.write("Add test points at which you want to <br> estimate the performance", unsafe_allow_html=True)
            st.write("#")
            st.write("#")
            df_test = st.experimental_data_editor(st.session_state.df_test, width=500, height=200, num_rows="dynamic")

            if st.button("Predict"):
                st.session_state.df_test = df_test.copy()
                st.session_state.df_test.reset_index(drop=True, inplace=True)

                # Find rows without missing values
                complete_rows_mask_test = st.session_state.df_test.notna().all(axis=1)

                # Check if all rows are complete
                if complete_rows_mask_test.all():
                    # Separate the features
                    test_x = st.session_state.df_test[complete_rows_mask_test].iloc[:, 1:]  # Exclude 'Point Index'
                    # Convert pandas DataFrames to PyTorch tensors
                    test_x = torch.tensor(test_x.values, dtype=torch.float32)
                    # Initialize the figure outside of the loop
                    fig = go.Figure()
                    layout = go.Layout(
                        xaxis=dict(title="Predicted Performance", title_font=dict(size=18), tickfont=dict(size=18)),
                        yaxis=dict(title="Probability Density", title_font=dict(size=18), tickfont=dict(size=18)),
                        autosize=False,
                        width=700,
                        height=400,
                        margin=go.layout.Margin(
                            l=50,
                            r=50,
                            b=100,
                            t=100,
                            pad=4))

                    # Make predictions for complete rows
                    for i, idx in enumerate(test_x):
                        input_tensor = torch.tensor(idx).reshape(1, -1)
                        mean, std_dev = st.session_state.BO.get_posterior_stats(input_tensor)

                        mean = mean.item()
                        std_dev = std_dev.item()
                        x_values = np.linspace(mean - 3*std_dev, mean + 3*std_dev, 1000)
                        fig.add_trace(go.Scatter(x=x_values, y=norm.pdf(x_values, mean, std_dev),
                                                    mode='lines',
                                                    name=f"Point {int(st.session_state.df_test.iloc[i, 0])}",
                                                    fill='tozeroy',
                                                    line=dict(color=color_list[i % len(color_list)])))  # Set the color of the trace

                    fig.update_layout(layout)
                    st.session_state.plot = fig
                else:
                    st.error("Please fill in all fields before predicting.")

        with preds_col2:
            st.markdown("##### Visualization of Performance")
            st.write("Estimated performance statistics based on <br> available observations", unsafe_allow_html=True)
            if 'plot' in st.session_state:
                st.write(st.session_state.plot)

        acquisition_col1 , acquisition_col2= st.columns([0.7,1.0])

        with acquisition_col1:
            st.markdown("#")
            st.markdown("##### Acquisition Inspection")
            st.write("Select a test point and a feature to inspect acquisition function value", unsafe_allow_html=True)

            if 'df_test' in st.session_state:
                # Create a selection box for the "Point Index"
                point_indices = [str(int(i)) for i in st.session_state.df_test['Index'].unique()]
                point_index = st.selectbox('Select Point Index', point_indices)
                point_index = int(point_index)
                # After the point index is selected by the user:
                display_row = st.session_state.df_test[st.session_state.df_test['Index'] == point_index].iloc[:, 1:]
                # Transpose the row to get a column DataFrame
                transposed_row = display_row.transpose()
                # Get column names (feature names) and values separately
                features = transposed_row.index.tolist()  # Feature names
                values = [v[0] for v in transposed_row.values.tolist()]  # Flatten the list and Corresponding values
                # Now, create a new DataFrame with 'Feature' and 'Value' columns, round values to 2 decimal places
                display_df = pd.DataFrame({'Feature': features, 'Value': [round(val, 2) for val in values]})

                # Convert the DataFrame to an HTML table string without index and header
                table_html = display_df.to_html(index=False, header=False)

                # Add CSS to the HTML string to change the font size
                table_html = '<div style="font-size: 14px">' + table_html + '</div>'

                # Display the HTML table string with Streamlit's markdown function
                st.markdown(table_html, unsafe_allow_html=True)

                st.markdown("#")

                #Select feature
                feature_name = st.selectbox('Select Feature', st.session_state.df_test.columns[1:].tolist())  # Exclude 'Point Index'

                # If the "Visualize" button is clicked
                if st.button('Visualize'):
                    # After selection, you can access the selected row and feature as follows:
                    selected_row = st.session_state.df_test[st.session_state.df_test['Index'] == point_index].iloc[:, 1:]  # Exclude 'Point Index'
                    selected_feature_value = selected_row[feature_name].values[0]

                    # Get the position of the selected feature
                    M = st.session_state.df_test.columns.get_loc(feature_name) - 1  # Subtract 1 because we excluded 'Point Index'

                    # Now, you can call the `predict_slice` method with the selected test_x and M
                    test_x = torch.tensor(selected_row.values, dtype=torch.float32)
                    N = 100  # For example, set N=100 to get a detailed prediction
                    xplot, means, std_devs, acqs = st.session_state.BO.predict_slice(test_x, M, acquisition_function=None, beta=2.0*0.5, N=N)
                    # You now have the predicted values for the selected feature over a range of its values, keeping all other features constant
                    with acquisition_col2:
                        # Create subplot with 2 rows
                        fig = make_subplots(rows=2, cols=1)

                        # Create mean and std_dev traces for upper plot
                        upper_trace = go.Scatter(x=xplot.numpy(), y=means.numpy(), mode='lines', name='Mean')
                        upper_trace_fill = go.Scatter(x=np.concatenate((xplot.numpy(), xplot.numpy()[::-1])),  # x, then x reversed
                                                      y=np.concatenate((means.numpy() - 1.96*std_devs.numpy(), (means+1.96*std_devs).numpy()[::-1])),  # upper, then lower reversed
                                                      fill='toself',
                                                      fillcolor='rgba(0,176,246,0.2)',
                                                      line=dict(color='rgba(255,255,255,0)'),
                                                      hoverinfo="skip",
                                                      showlegend=False)

                        # Add traces to the upper subplot
                        fig.add_trace(upper_trace, row=1, col=1)
                        fig.add_trace(upper_trace_fill, row=1, col=1)

                        # Create acquisition function trace for lower plot
                        lower_trace = go.Scatter(x=xplot.numpy(), y=acqs.numpy(), mode='lines', name='Acquisition Function')

                        # Add trace to the lower subplot
                        fig.add_trace(lower_trace, row=2, col=1)

                        # Update layout
                        fig.update_layout(height=600, width=700, title_text="Mean with Confidence Intervals and Acquisition Function", showlegend=False)

                        # Update xaxis and yaxis parameters for both subplots
                        fig.update_xaxes(title_text=feature_name, row=1, col=1)  # The x-axis label is the selected feature
                        fig.update_yaxes(title_text="Predicted Performance", row=1, col=1)
                        fig.update_xaxes(title_text=feature_name, row=2, col=1)  # The x-axis label is the selected feature
                        fig.update_yaxes(title_text="Acquisition Function", row=2, col=1)

                        # Display the figure
                        st.plotly_chart(fig)
    else:
       st.write("No file has been uploaded yet")
