import streamlit as st
import streamlit.components.v1 as stc

#import our app
from ml_app import run_ml_app

html_temp = """
            <div style="background-color:#3872fb;padding:10px;border-radius:10px">
		    <h1 style="color:white;text-align:center;">Hearth Attack Prediction</h1>
		    <h4 style="color:white;text-align:center;">Analytic Alience</h4>
		    </div>
            """

desc_temp = """
            ### Hearth Attack Prediction
            Aplikasi ini akan digunakan oleh rumah sakit dan puskesmas pada pasien yang memiliki riwayat dan resiko serangan jantung.
            #### Data Source
            - https://www.kaggle.com/datasets/mirzahasnine/heart-disease-dataset
            #### App Content
            - Machine Learning Section
            """

def main():
    # st.title("Main App")
    stc.html(html_temp)

    menu = ["Home","Machine Learning"]
    choice = st.sidebar.selectbox("Menu", menu) 
    
    # Ngarahin select box mau kemana aja
    if choice == "Home":
        # st.subheader("Home")
        st.markdown(desc_temp, unsafe_allow_html=True)
    elif choice == "Machine Learning":
        st.subheader("Machine Learning Section")
        run_ml_app()
    
# Ketika eksekusi maka akan eksekusi fuction main()
if __name__ == '__main__':
    main()
