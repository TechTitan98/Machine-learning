import streamlit as st
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import pickle as pkl
import google.generativeai as genai
st.title("INTRODUCTION TO PROJECT 5")
# st.code("st.write", language='python')
st.set_page_config(
    page_title="MY PROJECT 5",
    page_icon=":guardsman:",
    layout="wide",)

st.sidebar.title("PAGES")
# st.sidebar.write("") 
# st.caption('caption')
status = st.sidebar.radio("Select Page: ", ('NOTES','MONEY','supervised_learning','unsupervised_learning'))

def NOTES():
    st.title("ML(Machine Learning)ALGO(PREDICTION)")
    st.subheader("Machine Learning")
    st.text("Machine Learning is a subset of artificial intelligence that focuses on the development of algorithms\n" \
    "MACHINE LEARNING ALGORITHM ARE METHODS THAT ALLOW COMPUTERS TO LEARN PATTERNS FROM DATA AND MAKE PREDICTIONS OR DECISIONS WITHOUT BEING EXPLICITLY PROGRAMMED." \
    "\n\n"
    "1. SUPERVISED LEARNING: THIS ALGORITHM USES LABELED DATA TO TRAIN MODELS." \
    "\n"
    "2. UNSUPERVISED LEARNING: THIS ALGORITHM USES UNLABELED DATA TO FIND PATTERNS OR GROUPS." \
    "\n"
    "3. REINFORCEMENT LEARNING: THIS ALGORITHM LEARNS BY TRIAL AND ERROR" \
    "\n")
    # TYPE SELECTBOX
    type_choice = st.selectbox("Select Learning Type", ["None", "Supervised Learning", "Unsupervised Learning", "Reinforcement Learning"])
   
    supervised_algos = ["None", "Decision Tree", "KNN", "Linear Regression", "Logistic Regression", "Random Forest"]
    unsupervised_algos = ["None", "KMeans"]
    reinforcement_algos = ["None"]  # placeholder, you can add later

    if type_choice == "Supervised Learning":
        algo_choice = st.selectbox("Select Algorithm", supervised_algos)
    elif type_choice == "Unsupervised Learning":
        algo_choice = st.selectbox("Select Algorithm", unsupervised_algos)
    else:
        algo_choice = st.selectbox("Select Algorithm", reinforcement_algos)

    if type_choice == "Supervised Learning":
        st.markdown("label data means where input and output are designed(like in graph x and y are label)")
        st.write("1]supervised means learn form labeled data")
        st.write("2]there are two techniques in supervised,classification and regression")
        st.write("3]Classification :- in this , computer gives only two choice either 1 or 0, yes or no")
        st.write("4] Regression:- Continues value it is used when the you wants to computer predict number from to computer predict number from input or dataset" )

    if type_choice == "Unsupervised Learning":
        st.subheader("Unsupervised Learning")
        st.text("Unsupervised 📘: Unlabelled Data")

        st.markdown("### Techniques:")
        st.markdown("""
- **Scatterplot**
- **Clustering**: After clustering we get unlabelled data (e.g. only features, no output).
  If output is added later, then it becomes supervised.
        """)

        st.markdown("### Algorithms:")

        st.markdown("""
**1. K-Means (K = number of clusters)**  
- Groups similar or nearest data points.  
- Finds the mean of each group and uses it to form clusters.  
- Widely used for customer segmentation, document classification, etc.

**2. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**  
- Forms clusters based on data point density.  
- It combines closely packed data points into clusters.  
- Can detect outliers (noise) and works well with irregular shaped data.  
- Best for noisy or spatial datasets.
        """)

    elif type_choice == "Reinforcement Learning":
        st.markdown("""
                    - **No Data**  
                    - It works / learns from feedback of users and learns from previous moves.  
                    **Note:** DeepBlue is the first AI.
                    - **Two types**:  
                    - Q-Learning  
                    - SARSA (State Action Reward State Action)
                    - **Q-Learning**:  
                    - Uses Q-value  
                    - Maintains a Q-table
                    """)    


    if algo_choice == "Decision Tree":
        st.subheader("Decision Tree")
        st.markdown("""
                    **Decision Tree** is a supervised learning algorithm used for classification and regression.  
                    It splits the data into branches based on questions (conditions), leading to decisions (leaves).
                    """)
        st.code("""
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
""", language="python")
        st.image("decision tree.png", caption="Decision Tree Example", width=600)

    elif algo_choice == "KNN":
        st.subheader(" K-Nearest Neighbors (KNN)")
        st.markdown("""
                    **KNN** is a supervised learning algorithm used for classification and regression.  
                    It finds the 'K' closest data points to the test data and assigns the majority class.
                    """)
        st.code("""
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
""", language="python")
        st.image("KNN.jpg", caption="KNN Algorithm",width=600)

    elif algo_choice == "KMeans":
        st.subheader("K-Means Clustering")
        st.markdown("""
                    **KMeans** is an unsupervised learning algorithm used for clustering.  
                    It groups data into 'K' clusters based on similarity.
                    """)
        st.code("""
from sklearn.cluster import KMeans
model = KMeans(n_clusters=3)
model.fit(data)
labels = model.labels_
""", language="python")
        st.image("kmeans.png", caption="K-Means Clustering",width=600)

    elif algo_choice == "Linear Regression":
        st.subheader(" Linear Regression")
        st.markdown("""
                    **Linear Regression** is a supervised algorithm used to predict continuous values.  
                    It finds the best-fit line that maps input variables (X) to output (Y).
                    """)
        st.code("""
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
""", language="python")
        st.image("lr.png", caption="Linear Regression Line",width=600)

    elif algo_choice == "Random Forest":
        st.subheader("Random Forest")
        st.markdown("""
                    **Random Forest** is an ensemble method using multiple decision trees.  
                    It improves accuracy by reducing overfitting.
                    """)
        st.code("""
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
""", language="python")
        st.image("rf.png", caption="Random Forest Structure", width=600)

    elif algo_choice == "Logistic Regression":
        st.subheader(" Logistic Regression")
        st.markdown("""
                    **Logistic Regression** is used for binary classification (0 or 1).  
                    It uses the sigmoid function to output probabilities between 0 and 1.
                    """)
        st.code("""
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
""", language="python")
        st.image("logistic.jpg", caption="Logistic Regression Curve",width=600)  

def MONEY():
    st.title("🌍 LIVE CURRENCY CONVERTER")

    # List of currencies to choose from
    currency_dict = {
        'AED': 'United Arab Emirates Dirham',
        'AFN': 'Afghan Afghani',
        'ALL': 'Albanian Lek',
        'AMD': 'Armenian Dram',
        'ANG': 'Netherlands Antillean Guilder',
        'AOA': 'Angolan Kwanza',
        'ARS': 'Argentine Peso',
        'AUD': 'Australian Dollar',
        'AWG': 'Aruban Florin',
        'AZN': 'Azerbaijani Manat',
        'BAM': 'Bosnia-Herzegovina Convertible Mark',
        'BBD': 'Barbadian Dollar',
        'BDT': 'Bangladeshi Taka',
        'BGN': 'Bulgarian Lev',
        'BHD': 'Bahraini Dinar',
        'BIF': 'Burundian Franc',
        'BMD': 'Bermudian Dollar',
        'BND': 'Brunei Dollar',
        'BOB': 'Bolivian Boliviano',
        'BRL': 'Brazilian Real',
        'BSD': 'Bahamian Dollar',
        'BTN': 'Bhutanese Ngultrum',
        'BWP': 'Botswana Pula',
        'BYN': 'Belarusian Ruble',
        'BZD': 'Belize Dollar',
        'CAD': 'Canadian Dollar',
        'CDF': 'Congolese Franc',
        'CHF': 'Swiss Franc',
        'CLP': 'Chilean Peso',
        'CNY': 'Chinese Yuan',
        'COP': 'Colombian Peso',
        'CRC': 'Costa Rican Colón',
        'CUC': 'Cuban Convertible Peso',
        'CUP': 'Cuban Peso',
        'CVE': 'Cape Verdean Escudo',
        'CZK': 'Czech Koruna',
        'DJF': 'Djiboutian Franc',
        'DKK': 'Danish Krone',
        'DOP': 'Dominican Peso',
        'DZD': 'Algerian Dinar',
        'EGP': 'Egyptian Pound',
        'ERN': 'Eritrean Nakfa',
        'ETB': 'Ethiopian Birr',
        'EUR': 'Euro',
        'FJD': 'Fijian Dollar',
        'FKP': 'Falkland Islands Pound',
        'FOK': 'Faroese Króna',
        'GBP': 'British Pound Sterling',
        'GEL': 'Georgian Lari',
        'GGP': 'Guernsey Pound',
        'GHS': 'Ghanaian Cedi',
        'GIP': 'Gibraltar Pound',
        'GMD': 'Gambian Dalasi',
        'GNF': 'Guinean Franc',
        'GTQ': 'Guatemalan Quetzal',
        'GYD': 'Guyanese Dollar',
        'HKD': 'Hong Kong Dollar',
        'HNL': 'Honduran Lempira',
        'HRK': 'Croatian Kuna',
        'HTG': 'Haitian Gourde',
        'HUF': 'Hungarian Forint',
        'IDR': 'Indonesian Rupiah',
        'ILS': 'Israeli New Shekel',
        'IMP': 'Isle of Man Pound',
        'INR': 'Indian Rupee',
        'IQD': 'Iraqi Dinar',
        'IRR': 'Iranian Rial',
        'ISK': 'Icelandic Króna',
        'JEP': 'Jersey Pound',
        'JMD': 'Jamaican Dollar',
        'JOD': 'Jordanian Dinar',
        'JPY': 'Japanese Yen',
        'KES': 'Kenyan Shilling',
        'KGS': 'Kyrgyzstani Som',
        'KHR': 'Cambodian Riel',
        'KID': 'Kiribati Dollar',
        'KMF': 'Comorian Franc',
        'KRW': 'South Korean Won',
        'KWD': 'Kuwaiti Dinar',
        'KYD': 'Cayman Islands Dollar',
        'KZT': 'Kazakhstani Tenge',
        'LAK': 'Lao Kip',
        'LBP': 'Lebanese Pound',
        'LKR': 'Sri Lankan Rupee',
        'LRD': 'Liberian Dollar',
        'LSL': 'Lesotho Loti',
        'LYD': 'Libyan Dinar',
        'MAD': 'Moroccan Dirham',
        'MDL': 'Moldovan Leu',
        'MGA': 'Malagasy Ariary',
        'MKD': 'Macedonian Denar',
        'MMK': 'Burmese Kyat',
        'MNT': 'Mongolian Tögrög',
        'MOP': 'Macanese Pataca',
        'MRU': 'Mauritanian Ouguiya',
        'MUR': 'Mauritian Rupee',
        'MVR': 'Maldivian Rufiyaa',
        'MWK': 'Malawian Kwacha',
        'MXN': 'Mexican Peso',
        'MYR': 'Malaysian Ringgit',
        'MZN': 'Mozambican Metical',
        'NAD': 'Namibian Dollar',
        'NGN': 'Nigerian Naira',
        'NIO': 'Nicaraguan Córdoba',
        'NOK': 'Norwegian Krone',
        'NPR': 'Nepalese Rupee',
        'NZD': 'New Zealand Dollar',
        'OMR': 'Omani Rial',
        'PAB': 'Panamanian Balboa',
        'PEN': 'Peruvian Sol',
        'PGK': 'Papua New Guinean Kina',
        'PHP': 'Philippine Peso',
        'PKR': 'Pakistani Rupee',
        'PLN': 'Polish Złoty',
        'PYG': 'Paraguayan Guaraní',
        'QAR': 'Qatari Riyal',
        'RON': 'Romanian Leu',
        'RSD': 'Serbian Dinar',
        'RUB': 'Russian Ruble',
        'RWF': 'Rwandan Franc',
        'SAR': 'Saudi Riyal',
        'SBD': 'Solomon Islands Dollar',
        'SCR': 'Seychellois Rupee',
        'SDG': 'Sudanese Pound',
        'SEK': 'Swedish Krona',
        'SGD': 'Singapore Dollar',
        'SHP': 'Saint Helena Pound',
        'SLE': 'Sierra Leonean Leone (new)',
        'SLL': 'Sierra Leonean Leone (old)',
        'SOS': 'Somali Shilling',
        'SRD': 'Surinamese Dollar',
        'SSP': 'South Sudanese Pound',
        'STN': 'São Tomé and Príncipe Dobra',
        'SYP': 'Syrian Pound',
        'SZL': 'Eswatini Lilangeni',
        'THB': 'Thai Baht',
        'TJS': 'Tajikistani Somoni',
        'TMT': 'Turkmenistani Manat',
        'TND': 'Tunisian Dinar',
        'TOP': 'Tongan Paʻanga',
        'TRY': 'Turkish Lira',
        'TTD': 'Trinidad and Tobago Dollar',
        'TVD': 'Tuvaluan Dollar',
        'TWD': 'New Taiwan Dollar',
        'TZS': 'Tanzanian Shilling',
        'UAH': 'Ukrainian Hryvnia',
        'UGX': 'Ugandan Shilling',
        'USD': 'United States Dollar',
        'UYU': 'Uruguayan Peso',
        'UZS': 'Uzbekistani Soʻm',
        'VED': 'Venezuelan Digital Bolívar',
        'VES': 'Venezuelan Sovereign Bolívar',
        'VND': 'Vietnamese Đồng',
        'VUV': 'Vanuatu Vatu',
        'WST': 'Samoan Tala',
        'XAF': 'Central African CFA Franc',
        'XCD': 'East Caribbean Dollar',
        'XOF': 'West African CFA Franc',
        'XPF': 'CFP Franc (French overseas)',
        'YER': 'Yemeni Rial',
        'ZAR': 'South African Rand',
        'ZMW': 'Zambian Kwacha',
        'ZWL': 'Zimbabwean Dollar'
    }

    # Currency selection
    from_currency = st.selectbox("Convert From:", currency_dict.keys(), format_func=lambda x: f"{x} - {currency_dict[x]}")
    to_currency = st.selectbox("Convert To:", currency_dict.keys(), format_func=lambda x: f"{x} - {currency_dict[x]}")

    # Amount input
    amount = st.number_input("Enter Amount:", min_value=0.0, format="%.2f")

    if amount > 0 and from_currency and to_currency:
        if from_currency == to_currency:
            st.info("Same currencies selected. No conversion needed.")
        else:
            try:
                # Use your actual API key
                api_key = "65037c3e752b34875d9d38cb"
                
                # Construct correct URL
                url = f"https://v6.exchangerate-api.com/v6/{api_key}/latest/{from_currency}"
                
                # Send API request
                response = requests.get(url)
                data = response.json()

                if data["result"] == "success":
                    rate = data["conversion_rates"][to_currency]
                    converted_amount = amount * rate

                    # Show result
                    st.success(f"{amount} {from_currency} = {converted_amount:.2f} {to_currency}")
                    st.caption(f"🔁 Live Rate: 1 {from_currency} = {rate:.2f} {to_currency}")
                else:
                    st.error(f"API Error: {data.get('error-type', 'Unknown error')}")
            except Exception as e:
                st.error(f"❌ Error: {e}")



def supervised_learning():
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    import pickle
    import os

    st.title("📊 Project 5: Supervised Learning (Regression)")

    # Step 1: Load dataset
    df = pd.read_csv("Salary_dataset.csv")

    # Step 2: Clean Unnamed columns once and for all
    df_clean = df.loc[:,~df.columns.str.contains("^Unnamed", case=False)]

    # Step 3: Check & Drop nulls
    st.subheader("🔍 Missing Value Check")
    st.write(df_clean.isnull().sum())

    if df_clean.isnull().values.any():
        st.warning("⚠️ Null values found! Replacing with 1...")
        df_clean = df_clean.fillna(1)  # 👈 Yeh line sirf 1 se fill karega ✅
        st.success("✅ Null values filled with 1!")


    # Step 4: Show cleaned columns
    st.subheader("📋 Dataset Columns")
    st.markdown(f"🧮 Total Columns: **{len(df_clean.columns)}**")
    for col in df_clean.columns:
        st.markdown(f"- **{col}**")


    # Step 5: Show info using df_clean
    st.subheader("📋 Column Summary (Simple View):")
    for col in df_clean.columns:
        dtype = df_clean[col].dtype
        nulls = df_clean[col].isnull().sum()
        st.markdown(f"`Column:` {col} | `Type:` {dtype} | `Nulls:` {nulls}")

    # Step 6: Show describe
    st.subheader("📊 Summary Statistics")
    st.dataframe(df_clean.describe())

    # Step 7: Auto-select numeric features/target
    numeric_df = df_clean.select_dtypes(include=['int64', 'float64'])
    X = numeric_df.iloc[:, :-1]
    y = numeric_df.iloc[:, -1]

    # Step 8: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 9: Train models
    st.subheader("🤖 Training Models")

    model1 = LinearRegression()
    model1.fit(X_train, y_train)
    mse1 = mean_squared_error(y_test, model1.predict(X_test))

    model2 = RandomForestRegressor()
    model2.fit(X_train, y_train)
    mse2 = mean_squared_error(y_test, model2.predict(X_test))

    model3 = DecisionTreeRegressor()
    model3.fit(X_train, y_train)
    mse3 = mean_squared_error(y_test, model3.predict(X_test))

    model4 = KNeighborsRegressor()
    model4.fit(X_train, y_train)
    mse4 = mean_squared_error(y_test, model4.predict(X_test))

    # Step 10: Bar chart
    st.subheader("📉 Model MSE Comparison")
    models = ["Linear", "RandomForest", "DecisionTree", "KNN"]
    mses = [mse1, mse2, mse3, mse4]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(models, mses, color="skyblue")
    ax.set_ylabel("MSE")
    ax.set_title("Model Performance")
    st.pyplot(fig)

    # Step 11: Pick best
    mse_dict = {"Linear": mse1, "RandomForest": mse2, "DecisionTree": mse3, "KNN": mse4}
    best_model_name = min(mse_dict, key=mse_dict.get)
    st.write(f"🏆 Best model: {best_model_name}")

    # Step 12: Save best model
    best_model = {
        "Linear": model1,
        "RandomForest": model2,
        "DecisionTree": model3,
        "KNN": model4
    }[best_model_name]

    with open("best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    st.write("✅ Model saved as best_model.pkl")

    # Step 13: Predict with pickle
    with open("best_model.pkl", "rb") as f:
        loaded_model = pickle.load(f)
        pred = loaded_model.predict(X_test)

    # Step 14: Show actual vs predicted
    st.subheader("🔮 Actual vs Predicted")
    result = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": pred
    }).reset_index(drop=True)
    st.dataframe(result)

def unsupervised_learning():
    from sklearn.cluster import KMeans
    st.title("📈 Project 5: Unsupervised Learning (Clustering)")

    # Step 1: Load dataset
    df = pd.read_csv("Mall_Customers.csv")

    # Step 2: Remove CustomerID column
    if 'CustomerID' in df.columns:
        df.drop("CustomerID", axis=1, inplace=True)

    # Step 3: Clean unnamed columns (safety)
    df_clean = df.loc[:, ~df.columns.str.contains('^Unnamed', case=False)]

    # Step 4: Handle missing values
    st.subheader("🔍 Missing Value Check")
    st.write(df_clean.isnull().sum())

    if df_clean.isnull().values.any():
        st.warning("⚠️ Null values found! Replacing with 1...")
        df_clean = df_clean.fillna(1)
        st.success("✅ Nulls filled with 1!")

    # Step 5: Show columns
    st.subheader("📋 Dataset Columns")
    for col in df_clean.columns:
        st.text(f"Column: {col}")

    # Step 6: Info
    # Dataset Info (clean display)
    st.subheader("🧾 Dataset Info")
    buffer = io.StringIO()
    df_clean.info(buf=buffer)
    info_text = buffer.getvalue()
    st.markdown("```")
    st.markdown(info_text)
    st.markdown("```")


    # Step 7: Describe
    st.subheader("📊 Summary Statistics")
    st.dataframe(df_clean.describe())

    # Step 8: Use only numeric columns for clustering
    numeric_df = df_clean.select_dtypes(include=['float64', 'int64'])

    # Step 9: Apply KMeans
    st.subheader("🔗 KMeans Clustering")
    num_clusters = st.slider("Select number of clusters", 2, 10, 3)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(numeric_df)
    df_clean['Cluster'] = kmeans.labels_

    st.success(f"✅ Clustering completed with {num_clusters} clusters")

    # Step 10: Show clustered data
    st.subheader("📦 Clustered Data Sample")
    st.dataframe(df_clean.head())

    # Step 11: Cluster Plot
    st.subheader("🖼️ Cluster Plot (using 2 features)")

    if numeric_df.shape[1] >= 2:
        x_col = numeric_df.columns[0]
        y_col = numeric_df.columns[1]
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(
            data=df_clean,
            x=x_col,
            y=y_col,
            hue="Cluster",
            palette="Set2",
            s=100,
            ax=ax
        )
        plt.title(f"KMeans Clustering ({x_col} vs {y_col})")
        st.pyplot(fig)
    else:
        st.warning("❌ Need at least 2 numeric columns to plot clusters.")

if status == "NOTES":
    NOTES()

if status == "MONEY":
    MONEY()

if status == "supervised_learning":
    supervised_learning()

if status == "unsupervised_learning":
    unsupervised_learning()
