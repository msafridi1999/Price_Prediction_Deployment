import streamlit as st

# Title centered with custom color (e.g., blue) and removed black color
st.markdown("<h1 style='text-align: center; color: blue;'>Welcome to Price Prediction</h1>", unsafe_allow_html=True)

# Styled 'Choose a category' label in the sidebar
st.sidebar.markdown("<h3 style='color: green;'>Choose a category:</h3>", unsafe_allow_html=True)

# Navigation
page = st.sidebar.selectbox(
    "",
    ["Electronics Price Prediction", "Vehicles Price Prediction"]
)

# Electronics Navigation
if page == "Electronics Price Prediction":

   # Styled 'Choose a project' label in the sidebar
    st.sidebar.markdown("<h3 style='color: purple;'>Choose a project:</h3>", unsafe_allow_html=True)


    project = st.sidebar.selectbox(
        "",
        ["Laptop Price Prediction", "Mobile Price Prediction", "Television Price Prediction", "Camera Price Prediction"]
    )

    if project == "Laptop Price Prediction":
        # Laptop Predictor Code here
        import pickle
        import numpy as np

        # import the model
        pipe = pickle.load(open('pipe.pkl','rb'))
        df = pickle.load(open('df.pkl','rb'))

       # Display car image (you can replace the URL with a local image path if needed)
        st.image("/content/laptop-2575689_1280.jpg", width=700)

        st.title("Laptop Predictor") 

        # brand
        company = st.selectbox('Brand', df['Company'].unique())

        # type of laptop
        type = st.selectbox('Type', df['TypeName'].unique())

        # Ram
        ram = st.selectbox('RAM(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

        # weight
        weight = st.number_input('Weight of the Laptop')

        # Touchscreen
        touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

        # IPS
        ips = st.selectbox('IPS', ['No', 'Yes'])

        # screen size
        screen_size = st.selectbox('Scrensize (in inches)', [10.0, 11.6, 12.0, 13.0, 13.3, 14.0, 15.6, 17.0, 18.0])

        # resolution
        resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])

        #cpu
        cpu = st.selectbox('CPU', df['Cpu brand'].unique())

        hdd = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])

        ssd = st.selectbox('SSD(in GB)', [0, 8, 128, 256, 512, 1024])

        gpu = st.selectbox('GPU', df['Gpu brand'].unique())

        os = st.selectbox('OS', df['os'].unique())

        if st.button('Predict Price'):
            ppi = None
            if touchscreen == 'Yes':
                touchscreen = 1
            else:
                touchscreen = 0

            if ips == 'Yes':
                ips = 1
            else:
                ips = 0

            X_res = int(resolution.split('x')[0])
            Y_res = int(resolution.split('x')[1])
            ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size
            query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])

            query = query.reshape(1, 12)
            st.title("The predicted price of this configuration is " + str(int(np.exp(pipe.predict(query)[0]))))

    elif project == "Mobile Price Prediction":
        # Mobile Price Prediction code here
        import pickle
        import pandas as pd
        import numpy as np

        # Load the saved Linear Regression model and other required components
        lr_model = pickle.load(open('lr_model.pkl', 'rb'))
        ohe = pickle.load(open('ohe.pkl', 'rb'))  # Load the OneHotEncoder used during training
        data = pickle.load(open('data.pkl', 'rb'))

        # Display car image (you can replace the URL with a local image path if needed)
        st.image("/content/Mobile.jpg", width=700)

        # Streamlit UI
        st.title("Mobile Price Prediction App")
        st.header("Enter mobile specifications to predict the price")

        # Brand selection based on unique brand values used in the original dataset
        brand_options = list(data['Brand me'].unique())
        brand = st.selectbox("Brand", options=brand_options)

        # User inputs for mobile features
        ratings = st.number_input("Ratings (0-5)", min_value=0.0, max_value=5.0, value=4.0, step=0.1)
        ram = st.number_input("RAM (GB)", min_value=0, max_value=64, value=6)
        rom = st.number_input("ROM (GB)", min_value=0, max_value=512, value=64)
        mobile_size = st.number_input("Mobile Size (inches)", min_value=4.0, max_value=7.0, value=6.5)
        primary_cam = st.number_input("Primary Camera (MP)", min_value=2, max_value=108, value=48)
        selfi_cam = st.number_input("Selfie Camera (MP)", min_value=0, max_value=64, value=13)
        battery_power = st.number_input("Battery Power (mAh)", min_value=100, max_value=10000, value=4000)

        # Prepare the input data as a DataFrame
        input_data = pd.DataFrame({
            'Ratings': [ratings],
            'RAM': [ram],
            'ROM': [rom],
            'Mobile_Size': [mobile_size],
            'Primary_Cam': [primary_cam],
            'Selfi_Cam': [selfi_cam],
            'Battery_Power': [battery_power],
            'Brand me': [brand]
        })

        # Transform the 'Brand me' column using the trained OneHotEncoder
        encoded_brand = ohe.transform(input_data[['Brand me']])

        # Drop the 'Brand me' column and concatenate the encoded brand with other input features
        input_data = input_data.drop('Brand me', axis=1)
        input_data_encoded = pd.concat([input_data.reset_index(drop=True), 
                                        pd.DataFrame(encoded_brand, columns=ohe.get_feature_names_out())], axis=1)

        # Predict the price when the button is clicked
        if st.button("Predict Price"):
            predicted_price = lr_model.predict(input_data_encoded)  # Predict using the linear regression model
            st.title(f"The predicted price is ₹{predicted_price[0]:,.2f}")
            

    elif project == "Television Price Prediction":
         # Television Price Prediction code here
        import pandas as pd
        import numpy as np
        import pickle

        # Display car image (you can replace the URL with a local image path if needed)
        st.image("/content/TV image.jpg", width=700)

        # Set the title of the app
        st.title("📺 Television Price Prediction")

        # Add a brief description
        st.write(""" ### Enter the details of the television to predict its price. """)

        # Load the model
        pipe = pickle.load(open('pipe_model.pkl', 'rb'))
        df = pickle.load(open('df_tv.pkl', 'rb'))

        # Brand
        brand = st.selectbox('Brand', df['Brand'].unique())

        # Resolution
        resolution = st.selectbox('Resolution', df['Resolution'].unique())

        # Screen Size
        size = st.selectbox('Size (in inches)', df['Size'].unique())

        # Operating System
        operating_system = st.selectbox('Operating System', df['Operating System'].unique())

        # Rating
        rating = st.selectbox('Rating', df['Rating'].unique())

       # Predict button
        if st.button('Predict Price'):
          # Create a query array
          query = np.array([brand, resolution, size, operating_system, rating])
          query = query.reshape(1, 5)
    
          # Predict the price
          predicted_price = pipe.predict(query)[0]
    
          # Display the predicted price
          st.title(f"The predicted price of this configuration is ₹{int(predicted_price)}")
 
    elif project == "Camera Price Prediction":
        # Camera Price Prediction code here
        import pickle
        import pandas as pd
        import numpy as np
        # Load the pre-trained model and data encoders
        model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
        df = pickle.load(open('camera.pkl', 'rb'))

        # Display car image (you can replace the URL with a local image path if needed)
        st.image("/content/camera image.jpg", width=700)

        # Set the title of the Streamlit app
        st.title('Camera Price Prediction App')

        # Collect user input for each feature required for prediction
        def get_user_input():
            model = st.selectbox('Camera Model', df['Model'].unique())
            release_date = st.selectbox('Release Date', df['Release date'].unique())
            max_resolution = st.selectbox('Max Resolution', df['Max resolution'].unique())
            low_resolution = st.selectbox('Low Resolution', df['Low resolution'].unique())
            effective_pixels = st.selectbox('Effective Pixels (in MP)', df['Effective pixels'].unique())
            zoom_wide = st.selectbox('Zoom Wide (W)', df['Zoom wide (W)'].unique())
            zoom_tele = st.selectbox('Zoom Tele (T)', df['Zoom tele (T)'].unique())
            normal_focus_range = st.selectbox('Normal Focus Range', df['Normal focus range'].unique())
            macro_focus_range = st.selectbox('Macro Focus Range', df['Macro focus range'].unique())
            storage_included = st.selectbox('Storage Included (in MB)', df['Storage included'].unique())
            weight_inc_batteries = st.selectbox('Weight (including batteries) in grams', df['Weight (inc. batteries)'].unique())
            dimensions = st.selectbox('Dimensions (in mm)', df['Dimensions'].unique())

            # Store inputs into a dictionary
            user_data = {
                'Model': model,
                'Release date': release_date,
                'Max resolution': max_resolution,
                'Low resolution': low_resolution,
                'Effective pixels': effective_pixels,
                'Zoom wide (W)': zoom_wide,
                'Zoom tele (T)': zoom_tele,
                'Normal focus range': normal_focus_range,
                'Macro focus range': macro_focus_range,
                'Storage included': storage_included,
                'Weight (inc. batteries)': weight_inc_batteries,
                'Dimensions': dimensions
            }

            # Convert the user input to a DataFrame
            features = pd.DataFrame(user_data, index=[0])
            return features

        # Get user input
        user_input_df = get_user_input()

        # Predict the price
        if st.button('Predict Price'):
            prediction = model.predict(user_input_df)

            # Display the price in Indian Rupees (assuming the predicted price is in rupees already)
            st.subheader(f'Estimated Camera Price: ₹{prediction[0]:,.2f}')



# Vehicles Navigation
elif page == "Vehicles Price Prediction":
    project = st.sidebar.selectbox(
        "Choose a project",
        ["Car Price Prediction", "Bike Price Prediction"]
    )

    if project == "Car Price Prediction":
        # Car Price Prediction code here
       import pickle
       import pandas as pd
       import numpy as np

       # Load the model and the dataset
       model = pickle.load(open('LinearRegression.pkl', 'rb'))
       car = pd.read_csv('Cleaned_Car_data.csv')

       # Display car image (you can replace the URL with a local image path if needed)
       st.image("/content/car images.jpg", width=700)  # You can replace the URL with an actual image URL or path

       # Streamlit app title
       st.title("Car Price Predictor")

       # Input parameters on the main page (similar to the television predictor UI)
       company = st.selectbox('Company', car['company'].unique())

       car_model = st.selectbox('Car Model', car['name'].unique())

       year = st.selectbox('Year', sorted(car['year'].unique(), reverse=True))

       fuel_type = st.selectbox('Fuel Type', car['fuel_type'].unique())

       kms_driven = st.number_input('Kilometers Driven', min_value=0)

      # Button for making the prediction
       if st.button('Predict Price'):
         # Prepare the input query for prediction
          query = np.array([car_model, company, year, kms_driven, fuel_type])
          query = query.reshape(1, 5)
    
         # Predict the price using the trained model
          prediction = model.predict(pd.DataFrame(data=query, columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']))
    
         # Display the prediction result
          st.title(f"The predicted price of the car is ₹{np.round(prediction[0], 2)}")


    elif project == "Bike Price Prediction":
        # Bike Price Prediction code here
       import pickle
       import pandas as pd
       import numpy as np

       # Load the model and the dataset
       model = pickle.load(open('RandomForestRegressionModel.pkl', 'rb'))
       bike = pickle.load(open('bike.pkl', 'rb'))

       # Display car image (you can replace the URL with a local image path if needed)
       st.image("/content/Bike image.jpg", width=700) 

       # Clean column names by stripping extra spaces or tabs
       bike.columns = bike.columns.str.strip()

       # Streamlit app title
       st.title("Bike Price Predictor")

       # Input parameters on the main page (similar to the television predictor UI)
       company = st.selectbox('Company', bike['company'].unique())

       Engine_warranty = st.selectbox('Engine Warranty', bike['Engine_warranty'].unique())

       Engine_type = st.selectbox('Engine Type', bike['Engine_type'].unique())

       Fuel_type = st.selectbox('Fuel Type', bike['Fuel_type'].unique())

       Cubic_capacity = st.selectbox('Cubic Capacity (in cc)', bike['Cubic_capacity'].unique())

       Fuel_capacity = st.selectbox('Fuel Capacity (in liters)', bike['Fuel_Capacity'].unique())

       # Button for making the prediction
       if st.button('Predict Price'):
           # Predict the price using the trained model
           query = np.array([company, Engine_warranty, Engine_type, Fuel_type, Cubic_capacity, Fuel_capacity])
           query = query.reshape(1, 6)
    
           prediction = model.predict(pd.DataFrame(columns=['company', 'Engine_warranty', 'Engine_type', 'Fuel_type', 'Cubic_capacity', 'Fuel_Capacity'], data=query))
    
           # Show the prediction result
           st.title(f"The predicted price of the bike is ₹{np.round(prediction[0], 2)}")