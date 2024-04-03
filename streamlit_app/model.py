import streamlit as st
import pickle

#fancy stuff
st.set_page_config(
        page_title="Movie Review",
        page_icon="🎞️",#takes emojies
        layout="centered",
        initial_sidebar_state="expanded",
)

vectorizer=pickle.load(open(r"C:\Users\admin\Documents\Current_Workspace\Programming\SVM\Pickle\Main Models\vectorizer_tfid.pkl","rb"))
model_SVM = pickle.load(open(r"C:\Users\admin\Documents\Current_Workspace\Programming\SVM\Final_pickles\classifier_SVM.pkl","rb"))
model_NaiveBayes= pickle.load(open(r"C:\Users\admin\Documents\Current_Workspace\Programming\SVM\Final_pickles\classifier_NaiveBayes.pkl","rb"))
model_dictionary={'SVM':model_SVM,
                  'Naive Bayes':model_NaiveBayes}
SVM_piechart=pickle.load(open(r"C:\Users\admin\Documents\Current_Workspace\Programming\SVM\Final_pickles\svm_pie.pkl","rb"))
NB_piechart=pickle.load(open(r"C:\Users\admin\Documents\Current_Workspace\Programming\SVM\Final_pickles\nb_pie.pkl","rb"))
chart_dict={'SVM':SVM_piechart,
            'Naive Bayes':NB_piechart}
#test fuction
def Run_models(Input:str,model,vectorizer):
    return model.predict(vectorizer.transform([Input].__iter__()))

st.title("Movie review sentiment 🎞️🍿🎥")

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Input columns
st.sidebar.header("Select Model")
st.header("Enter the movie review here")
Input = st.text_input('')
model_selection= st.sidebar.selectbox("Choose a Model", ["SVM","Naive Bayes"])

# enter your output here
if st.button("Check response"):
    sentiment=Run_models(Input,model_dictionary[model_selection],vectorizer)
    # st.success(f'The review is {"positive✅" if sentiment else "negative"}')
    if sentiment:
        st.success(f'The review is positive✅')
    else:
        st.error(f'The review is negative❌')

if st.button("Show Pie Chart"):
    fig=chart_dict[model_selection]
    st.plotly_chart(fig)
