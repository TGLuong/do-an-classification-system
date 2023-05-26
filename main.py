import streamlit as st
import time
import string
from Lib.AndroidDetection import AndroidDetection
from Lib.library import save_uploaded_file, decompile, load_pkl, get_api

model = AndroidDetection()
api_dataset = load_pkl("Res/api_dataset.pkl")

if __name__ == '__main__':
    st.set_page_config(page_title='APK Malware Detection')
    st.sidebar.image('LogoHVKTMM.png')
    st.sidebar.title("Upload APK File")
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.header('*Android Malware Classification System*')
    st.markdown("The system classify android application into five type of label is **_Adware_**, **_Banking_**, \
                **_Benign_**, **_Ristware_** and **_SMSMalware_**")
    st.divider()
    uploaded_file = st.sidebar.file_uploader(" ")
    if uploaded_file:
        file_path = save_uploaded_file(uploaded_file)
        raw_data = get_api(file=file_path, api_dataset=api_dataset)
        st.info("Extract file OK")
        st.divider()
        progress_text = "PREDICTION"
        my_bar = st.progress(0, text=progress_text)

        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text=progress_text)
        response = model.predict_raw_data(raw_data)
        st.success("{}".format(response["label"].upper()))
        st.divider()

        
    else:
        st.sidebar.write("Please upload an apk file")
