import streamlit as st
import pandas as pd
import requests
import os
import io
import cv2
import requests
import numpy as np
from IPython.display import Image, display

base_url = 'http://0.0.0.0:8000'
endpoint = '/predict'
model = 'yolov3-tiny'

url_with_endpoint_no_params = base_url + endpoint
url_with_endpoint_no_params

full_url = url_with_endpoint_no_params + "?model=" + model
full_url

st.title('Ml Models Dashboard')

# Sidebar
page = st.sidebar.selectbox('Page Navigation', ["object-detection", "."],key='page_nav')

def response_from_server(url, image_file, verbose=True):
    """Makes a POST request to the server and returns the response.

    Args:
        url (str): URL that the request is sent to.
        image_file (_io.BufferedReader): File to upload, should be an image.
        verbose (bool): True if the status of the response should be printed. False otherwise.

    Returns:
        requests.models.Response: Response from the server.
    """
    
    files = {'file': image_file}
    response = requests.post(url, files=files)
    status_code = response.status_code
    if verbose:
        msg = "Everything went well!" if status_code == 200 else "There was an error when handling the request."
        print(msg)
    return response

def display_image_from_response(response):
    """Display image within server's response.

    Args:
        response (requests.models.Response): The response from the server after object detection.
    """
    
    image_stream = io.BytesIO(response.content)
    image_stream.seek(0)
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(image, use_column_width=True)


def main():
    if st.session_state.page_nav =='object-detection':
        st.title('Object Detection')
        st.file_uploader(label='Upload An Image',type='.png',key='image_file')
        if st.session_state.image_file is not None:
            prediction = response_from_server(full_url, st.session_state.image_file)
            display_image_from_response(prediction)
        
main()