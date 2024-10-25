# Uncomment the following line if you do not have the Python module 'librosa' installed
# !pip install librosa

import os
import numpy as np
import pandas as pd
import librosa

# Function to create MFCC coefficients given an audio file

def create_MFCC_coefficients(file_name):

    sr_value = 44100
    n_mfcc_count = 20
    
    try:
        # Load the audio file using librosa
        y, sr = librosa.load(file_name, sr=sr_value)
              
        # Compute MFCC coefficients for the segment
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc_count)
        #print(f"file_name: {file_name}: y={len(y)}, sr={sr}, mfccs matrix:{np.shape(mfccs)}")
        
        # Create and return MFCC dataframe
        coeff_df = pd.DataFrame(mfccs)
        
        return coeff_df

    except Exception as e:
       print(f"Error creating MFCC coefficients: {file_name}:{str(e)}")