import streamlit as st
import pickle
from PIL import Image

#st.set_page_config(layout='wide')

loaded_vectorizer=pickle.load(open('vectorizer.sav','rb'))

loaded_model=pickle.load(open('spam_mail_model.sav','rb'))

spam_img=Image.open('spam.jpg')

def spam_prediction(input_data):

    #input_data = ["Hey amujuri,As your mentor,i'm excited to share incredible news with you  Brace yourself for an remarkable journey as coding as ningas proudly presentys its first ever batch of java"]

    input_data_feature_extraction = loaded_vectorizer.transform(input_data)

    final_prediction =loaded_model.predict(input_data_feature_extraction)

    

    if final_prediction[0]==1:
        
        return 'it is a "HAM" mail'
    else:
        st.image(spam_img , width=200)
        return 'it is a "SPAM" mail'
        

def main():
    st.title('SPAM MAIL PREDICTION APP')

    mail_text=st.text_area('Enter your mail data')

    final_result=''

    col1,col2,col3=  st.columns([0.26, 0.3, 0.1])

    if st.button('mail result'):
        final_result=spam_prediction([mail_text])
    
    st.text(final_result)



if __name__=='__main__':
    main()

