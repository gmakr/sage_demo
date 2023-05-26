import os
import streamlit as st
import pandas as pd
from tabulate import tabulate
from PIL import Image
import torch
import numpy as np
import plotly.graph_objects as go
import time
# Import the ExcelHandler class from your Jupyter Notebook
from single_utils import ExcelHandler, BayesianOptimization, GPModel
import datetime
from scipy.stats import norm

# Set page title and favicon
st.set_page_config(
    page_title="SAGE",
    page_icon=":robot_face:",
    layout="wide",
)


empty1, empty2, col1 = st.columns([1.6,1.6,0.5])  # creating empty slots

# Display the logo in the last column
# Load logo
logo_image = Image.open("utils/logo_small.png")
col1.image(logo_image,width=150)

# Display logo and prompt user to upload Excel file
#st.image(logo_image, width=300)
#st.title("Bayesian Optimization for Smart Experimentation (BOSE)")
#st.subheader('BOSE is an application for performing active experimental design based on Bayesian Optimization.')

tab_about, tab_data, tab_opt, tab_preds = st.tabs(["ℹ️ About", "🗃Data  ", "  📈 Optimization ", " 🎯 Predictions"])

with tab_about:
    st.markdown(
        """
        ## 💡 What is SAGE?
        Our app, "Smart Adaptive Guidance for Experiments" (SAGE), is a suite for active
        design of experiments. By harnessing the power of the latest advancements in machine learning,
        it solves optimization problems that are complex and expensive to evaluate. Designed to be
        user-friendly and intuitive, it ensures that users, regardless of their technical background,
        can easily use the application to its full potential.

        ## 🖥 How it works?
        The app consists of three main tabs, each with its unique functionality:

        - **🗃Data**: This is where you upload your initial experimental data. The data should be in an Excel format. Please make
        sure you follow the "template.xslm" file provided and include both the experimental parameters (features) and the observed results (targets).
        - **📈 Optimization**: In this tab, you'll see the optimization process in action. The system will suggest the next set of experiment parameters to try based on the current available data. It uses a Bayesian Optimization approach to intelligently suggest the next experiments.
        The tool provides the flexibility to insert any query that you perform.
        - **🎯 Predictions**: Here, you can input a set of parameters and the app will give you an estimate of the expected result, along with a probability distribution around that estimate. It allows you to explore the potential outcomes of an experiment without having to actually perform it.

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

        opt_col1, opt_col2 = st.columns([0.9,1.1])

        st.markdown("""
            <style>
            .stButton>button {
                background-color: lightgray;
            }
            </style>
            """, unsafe_allow_html=True)

        with opt_col1:
            # Display the dataframe
            st.markdown("#### Experimental Data")
            st.write("Add completed queries here")
            st.markdown('###')



            # Check if the "Index" column already exists
            if "Index" not in st.session_state.df_train.columns:
                # Generate the index
                index_values = list(range(-st.session_state.Ninit, 0)) + list(range(1, len(st.session_state.df_train) - st.session_state.Ninit + 1))
                # Insert the index as the first column
                st.session_state.df_train.insert(0, "Index", index_values)

            # Run the data editor
            st.session_state.df_train = st.experimental_data_editor(st.session_state.df_train, width=500, height=250, num_rows="dynamic")

            # Update the "Index" column with new values
            #index_values = list(range(-st.session_state.Ninit, 0)) + list(range(1, len(st.session_state.df_train) - st.session_state.Ninit + 1))
            #st.session_state.df_train["Index"] = index_values

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

            if st.button("Get Experiment"):
                st.markdown("#### Suggested Query")
                st.write("Queue of proposed experiment(s)")
                # Update the BO object
                st.session_state.BO = BayesianOptimization(train_X=train_x, train_Y=train_y, bounds=bounds_tensor, noiseless_obs=False)
                suggested_point, acq_value = st.session_state.BO.optimize_acquisition()

                # Convert suggested point to numpy array
                suggested_point_np = suggested_point.detach().numpy().flatten()

                # Construct a DataFrame from suggested point with appropriate column names
                st.session_state.suggested_df = pd.DataFrame([suggested_point_np], columns=col_names_x)

                # Add "Value" as column header
                st.session_state.suggested_df = st.session_state.suggested_df.rename_axis("Value", axis=1)

            # Display the stored suggested_df if it exists
            if hasattr(st.session_state, 'suggested_df'):
                table = tabulate(st.session_state.suggested_df, tablefmt="pipe", headers="keys", showindex=False)
                st.markdown(table, unsafe_allow_html=True)

        with opt_col2:
            # Get the dataframe from the session state
            df_train = st.session_state.df_train

            # Separate the target variable (y) from the features (x)
            df_train_y = df_train.iloc[:, -1]  # assuming y is the last column

            # Create a list for the x-axis
            x_axis = list(range(-st.session_state.Ninit, 0)) + list(range(1, df_train.shape[0]-st.session_state.Ninit+1))

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
                # Update the BO object in the session state
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
                    name='Predicted Performance' if i == st.session_state.Ninit else "",
                    legendgroup="group2",
                    hovertemplate="Predicted Performance at Iteration %d" % (i-st.session_state.Ninit+1),
                    showlegend=False if i > st.session_state.Ninit else True
                ))

                # Check if the actual observation is available
                if not np.isnan(df_train_y[i]):
                    # Add trace for actual observation
                    fig.add_trace(go.Scatter(
                        x=x_axis[i:i+1],
                        y=df_train_y[i:i+1],
                        mode='markers',
                        marker=dict(color='mediumvioletred', symbol='star', size=20),
                        name='Observed Performance' if i == st.session_state.Ninit else "",
                        legendgroup="group3",
                        hovertemplate="Actual Performance at Iteration %d" % (i-st.session_state.Ninit+1),
                        showlegend=False if i > st.session_state.Ninit else True
                    ))

            # Positioning the legend and adding axis names
            fig.update_layout(
                xaxis=dict(
                    title="Iterations",
                    titlefont=dict(
                        size=14 ),
                    tickfont=dict(
                        size=14),
                    showgrid = True
                ),
                yaxis=dict(
                    title="Performance",
                    titlefont=dict(
                        size=14
                    ),
                    tickfont=dict(
                        size=14
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



with tab_preds:
    if uploaded_file is not None:
        st.header("Predict at Desired Input")

        col1, col2 = st.columns([0.8,1.2])

        with col1:
            st.markdown("Estimate performance statistics based on <br> available observations", unsafe_allow_html=True)
            st.markdown("#")
            # Define a dictionary to hold user input
            input_dict = {}

            # Loop over each feature and get user input
            for feature in col_names_x:
                suggested_value = float(st.session_state.suggested_df[feature][0]) if hasattr(st.session_state, 'suggested_df') else 0.0
                input_dict[feature] = st.number_input(f"{feature}", value=suggested_value)


            # Convert the input dictionary to a DataFrame
            df_input = pd.DataFrame([input_dict])

            st.session_state.prediction_clicked = st.button("Predict")
            if st.session_state.prediction_clicked:
                # Convert the input dataframe to a tensor
                input_tensor = torch.tensor(df_input.values, dtype=torch.float32)
                # Exclude the last data point
                #train_x_excluding_last = train_x[:]
                #train_y_excluding_last = train_y[:]

                # Create the BO object using the modified data
                st.session_state.BO = BayesianOptimization(train_X=train_x,
                                                        train_Y=train_y,
                                                        bounds=bounds_tensor,
                                                        noiseless_obs=False)

                # Get the posterior stats for the input
                mean, std_dev = st.session_state.BO.get_posterior_stats(input_tensor)

                # Since mean and std_dev are tensors, we should convert them to scalar values
                st.session_state.mean_scalar = mean.item()
                st.session_state.std_dev_scalar = std_dev.item()

        with col2:
            if st.session_state.prediction_clicked:

                # Create an array of x values from the mean - 3*std_dev to mean + 3*std_dev
                x_values = np.linspace(st.session_state.mean_scalar - 3*st.session_state.std_dev_scalar,
                                       st.session_state.mean_scalar + 3*st.session_state.std_dev_scalar, 200)
                # Use the probability density function (pdf) to get y-values
                y_values = norm.pdf(x_values, st.session_state.mean_scalar, st.session_state.std_dev_scalar)

                # Create a trace for the normal distribution
                trace = go.Scatter(x=x_values, y=y_values, mode='lines', fill='tozeroy',
                                   line=dict(color='lightblue', width=2), name='Prediction')

                # Create a layout for the plot
                layout = go.Layout(
                    title="Model Predictions",
                    xaxis=dict(title="Predicted Performance",title_font=dict(size=18),tickfont=dict(size=18)),
                    yaxis=dict(title="Probability Density",title_font=dict(size=18),tickfont=dict(size=18)),
                    autosize=False,
                    width=500,
                    height=500,
                    margin=go.layout.Margin(
                        l=50,
                        r=50,
                        b=100,
                        t=100,
                        pad=4))

                # Add the trace to the plot
                fig = go.Figure(data=trace, layout=layout)

                # Show the plot
                st.plotly_chart(fig)

    else:
        st.write("No file has been uploaded yet.")
