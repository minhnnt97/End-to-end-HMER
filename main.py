import streamlit as st
import SessionState
from streamlit_drawable_canvas import st_canvas
from translate import *


PROJECT = '/Users/MAC/Desktop/Minh Nguyen/CoderSchool/HMER'
MODELS = os.path.join(PROJECT, 'models')
DEMO   = os.path.join(PROJECT, 'demo')

@st.cache(allow_output_mutation=True)
def load_models(model_name):
    enc_dir = os.path.join(MODELS, model_name, 'encoder')
    dec_dir = os.path.join(MODELS, model_name, 'decoder')

    enc = tf.keras.models.load_model(enc_dir, compile=False)
    dec = tf.keras.models.load_model(dec_dir, compile=False)

    return enc, dec


# MODEL LOADING
model_name = 'epoch_40_resize'
encoder, decoder = load_models(model_name)

    


##########################################################
###################### STREAMLIT UI ######################
##########################################################

# Create a side menu 
menu = ['Image']
choice = st.sidebar.selectbox('NAVIGATION', menu)

ss = SessionState.get(choice='Home', translate_flag=False, drawn_flag=False)
def reset_session_state():
    ss.translate_flag = False
    ss.drawn_flag = False

if ss.choice != choice:
    reset_session_state()
    ss.choice = choice

# Create the Home page
#if choice == 'Home':
#    st.header('HOME PAGE')


## Create the First page
#elif choice == 'Canvas':
#    st.header('CANVAS-2-LATEX')
#
#    canvas_result = st_canvas(
#        stroke_width=4,
#        stroke_color='#000000',
#        background_color='#ffffff',
#        background_image=None,
#        update_streamlit=False,
#        height=300,
#        width=484,
#        drawing_mode='freedraw',
#        key="canvas",
#        display_toolbar=True
#    )
#
#    drawn_flag = st.button('Translate')
#    if drawn_flag:
#        ss.drawn_flag = True
#
#    if canvas_result.image_data is not None:
#        if ss.drawn_flag:
#
#            with st.spinner('Translating...'):
#                img = preprocess_st(canvas_result.image_data)
#                result, attention = translate(img, encoder, decoder)
#
#            st.markdown('LaTeX translation:')
#            result_str = ' '.join(result[1:-1])
#            st.text('$ ' + result_str + ' $')
#            try:
#                st.latex(result_str)
#            except:
#                st.error('Could not display translation in latex.')
#
#            plot_attention_flag = st.button('Plot Model Attention')
#            if plot_attention_flag:
#                with st.spinner('Plotting...'):
#                    st.pyplot(plot_attention(img, result, attention))
#    else:
#        if ss.drawn_flag:
#            ss.drawn_flag=False


# Create the Second page
if choice == 'Image':
    st.header('IMAGE-2-LATEX')

    # Image uploader
    image_file = st.file_uploader("Upload Your Math Expression", type=['png','jpg'], key='image_upload')
    
    if image_file != None:
        # Get image
        user_image = decode_img(image_file)
        st.image(user_image, use_column_width="always")

        translate_flag = st.button('Translate')
        if translate_flag:
            ss.translate_flag=True
        
        if ss.translate_flag:
            with st.spinner('Translating...'):
                img = preprocess_st(user_image, uploaded=True)
                result, attention = translate(img, encoder, decoder)

            st.markdown('LaTeX translation:')
            result_str = ' '.join(result[1:-1])
            st.text('$ ' + result_str + ' $')
            try:
                st.latex(result_str)
            except:
                st.error('Could not display translation in latex.')

            plot_attention_flag = st.button('Plot Model Attention')
            if plot_attention_flag:
                with st.spinner('Plotting...'):
                    st.pyplot(plot_attention(img, result, attention))
    elif ss.translate_flag:
        ss.translate_flag = False
