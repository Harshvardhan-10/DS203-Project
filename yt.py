import os
import numpy as np
import pandas as pd
import librosa
import yt_dlp
import warnings

# Suppress the specific warning from librosa
warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")

# Ensure the 'songs' directory exists
output_folder = 'songs'
os.makedirs(output_folder, exist_ok=True)

def download_and_convert_to_mfcc(youtube_url, index, label, output_path=output_folder):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_path, f'song_{index}.%(ext)s'),
        'noplaylist': True,  # Download single video
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=True)
            audio_file_path = os.path.join(output_path, f"song_{index}.{info_dict['ext']}")
            song_name = info_dict.get('title', f'song_{index}')

            try:
                # Calculate MFCC coefficients
                coeff_df = create_MFCC_coefficients(audio_file_path)
                if coeff_df is not None:
                    coeff_csv_path = os.path.join(output_path, f"song_{index}_mfcc.csv")
                    coeff_df.to_csv(coeff_csv_path, index=False)
                    print(f"MFCC coefficients saved as: {coeff_csv_path}")
                    # Assign a label based on the index (grouped by 10)
                    # Set pathname to include the 'songs' folder
                    pathname = os.path.join('songs', f"song_{index}_mfcc.csv")
                    return {
                        'name': song_name,
                        'label': label,
                        'pathname': pathname
                    }
                else:
                    print(f"MFCC generation failed for {audio_file_path}")
                    return None

            finally:
                # Ensure the audio file is deleted regardless of success or failure
                if os.path.exists(audio_file_path):
                    os.remove(audio_file_path)

    except Exception as e:
        print(f"An error occurred while processing the video: {e}")
        return None

# Function to create MFCC coefficients given an audio file
def create_MFCC_coefficients(file_name):
    sr_value = 44100
    n_mfcc_count = 20
    
    try:
        # Load the audio file using librosa
        y, sr = librosa.load(file_name, sr=sr_value)

        # Compute MFCC coefficients for the segment
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc_count)

        # Create and return MFCC dataframe
        coeff_df = pd.DataFrame(mfccs)
        
        return coeff_df

    except Exception as e:
        print(f"Error creating MFCC coefficients: {file_name}: {str(e)}")
        return None

# List of YouTube URLs
urls = [
        # Michael Jackson
        "https://www.youtube.com/watch?v=oRdxUFDoQe0",
        "https://www.youtube.com/watch?v=QY9M-LrgotM",
        "https://www.youtube.com/watch?v=fq8IZninxio",
        "https://www.youtube.com/watch?v=sOnqjkJTMaA",
        "https://www.youtube.com/watch?v=H1cELUetPFM",
        "https://www.youtube.com/watch?v=dsUXAEzaC3Q",
        "https://www.youtube.com/watch?v=yURRmWtbTbo",
        "https://www.youtube.com/watch?v=Hqsslwcb3Qg",
        "https://www.youtube.com/watch?v=lDnz7oJv18E",
        "https://www.youtube.com/watch?v=QOnABvJWS30&list=PLAKPPAHY4SIKwNBbAcVa48KVUBbh6PHOA&index=5",
        'https://youtu.be/Zi_XLOBDo_Y',
        'https://youtu.be/QNJL6nfu__Q?si=28CZMObcQbtrofkq',
        'https://youtu.be/YP3W-E0OamU?si=MTSakxOmu7eE4RfS',
        'https://youtu.be/8GB9BULxZ8c?si=dv4-QeooiVkmXJ04',
        'https://youtu.be/tKc3VcOyY2c?si=xT7d_LqtuEAOING2',
        'https://youtu.be/csARzcsjark?si=VYvLqc1QwQlOcZDF',
        'https://youtu.be/kRp_FqCmsVA?si=tiQKBj8Myds6saD2',
        'https://youtu.be/jQY_QL_wvQU?si=vFn2WZSiJjXnwOL1',
        'https://youtu.be/ElN_4vUvTPs?si=3SA-GU5y4E3f3GvG',
        'https://youtu.be/RTXITtbKGNk?si=_3GInvQxV3iDarUg',
        # Asha Bhosale
        "https://www.youtube.com/watch?v=shcd0ZhUUYE&list=PLBfVktgJUq8wn3F70CnoqBjlMJKyx8ATE",
        "https://www.youtube.com/watch?v=V3lgVtktS5U&list=PLBfVktgJUq8wn3F70CnoqBjlMJKyx8ATE&index=4",
        "https://www.youtube.com/watch?v=4nwBlCsvwJs&list=PLBfVktgJUq8wn3F70CnoqBjlMJKyx8ATE&index=6",
        "https://www.youtube.com/watch?v=I1NAynlzRMA&list=PLBfVktgJUq8wn3F70CnoqBjlMJKyx8ATE&index=10",
        "https://www.youtube.com/watch?v=F2w5fH9ArkY&list=PLBfVktgJUq8wn3F70CnoqBjlMJKyx8ATE&index=12",
        "https://www.youtube.com/watch?v=HmoIL3UuEHk&list=PLBfVktgJUq8wn3F70CnoqBjlMJKyx8ATE&index=23",
        "https://www.youtube.com/watch?v=CigHzJ9p6hg&list=PLBfVktgJUq8wn3F70CnoqBjlMJKyx8ATE&index=24",
        "https://www.youtube.com/watch?v=jK_58xsBeo8&list=PLBfVktgJUq8wn3F70CnoqBjlMJKyx8ATE&index=28",
        "https://www.youtube.com/watch?v=9d--a7DG5vo&list=PLBfVktgJUq8wn3F70CnoqBjlMJKyx8ATE&index=31",
        "https://www.youtube.com/watch?v=FAPDbyDBNy8&list=PLBfVktgJUq8wn3F70CnoqBjlMJKyx8ATE&index=47",
        'https://youtu.be/d2fP18bGigs?si=jIm1iojhlhNAt_xF',
        'https://youtu.be/2nUsr94hyXk?si=T3PJJ-BcePCp1d2j',
        'https://youtu.be/A7AUKyAp7gk?si=YjSKNy6rHq_loqpX',
        'https://youtu.be/9TRVrA9Znsk?si=l9-h6N0OcOGQ2kDz',
        'https://youtu.be/tCEj3yJ2whM?si=V1W7DwMZ-YFiMabW',
        'https://youtu.be/Su4vvd2thG8?si=yZkDXiiR-RHmvSQD',
        'https://youtu.be/Np13tF0uDCo?si=5sNbiM1B2c3rHs25',
        'https://youtu.be/NrFWAokXglg?si=YU1DQ1JnAu2SRpd_',
        'https://youtu.be/mW3rc51kEUA?si=UPgZ24Yfdga5iCCx',
        'https://youtu.be/w716jLnQpIs?si=0MzFR38eXI1vChDV',
        # Kishore Kumar
        "https://www.youtube.com/watch?v=FFbc-jXkADs&list=PLUOEf-vLOCSkxWY5z9cjS4OT3oZ9D8suk",
        "https://www.youtube.com/watch?v=j7TM2ccOGbU&list=PLUOEf-vLOCSkxWY5z9cjS4OT3oZ9D8suk&index=2",
        "https://www.youtube.com/watch?v=lbfWsIpXsCA&list=PLUOEf-vLOCSkxWY5z9cjS4OT3oZ9D8suk&index=3",
        "https://www.youtube.com/watch?v=vhZLopg5kP0&list=PLUOEf-vLOCSkxWY5z9cjS4OT3oZ9D8suk&index=6",
        "https://www.youtube.com/watch?v=9PdSmDRGIwM&list=PLUOEf-vLOCSkxWY5z9cjS4OT3oZ9D8suk&index=7",
        "https://www.youtube.com/watch?v=daxuKHBWiKE&list=PLUOEf-vLOCSkxWY5z9cjS4OT3oZ9D8suk&index=30",
        "https://www.youtube.com/watch?v=-OUiQFr4j1k&list=PLUOEf-vLOCSkxWY5z9cjS4OT3oZ9D8suk&index=43",
        "https://www.youtube.com/watch?v=yIzCBU0_LyY&list=RDEMa_EYmHyXjUx2YUwQ3j4-UQ&start_radio=1",
        "https://www.youtube.com/watch?v=S0WPSYFm7iE&list=RDEMa_EYmHyXjUx2YUwQ3j4-UQ&index=4",
        "https://www.youtube.com/watch?v=69pPYkGiEAQ&list=RDEMa_EYmHyXjUx2YUwQ3j4-UQ&index=7",
        'https://youtu.be/H7Y5a7Y-jng?si=GL-i6rMxWn_kv33e',
        'https://youtu.be/KSKd4vIj0xk?si=-qZWsy7WrUKQV55z',
        'https://youtu.be/kBKA3g8WTuE?si=rXovxX-3NPmPZ0Qj',
        'https://youtu.be/dyEdcOhxJNQ?si=HfggsHzOimywtZus',
        'https://youtu.be/4CwFFWleNNA?si=xvgSKAPX8jM_R32N',
        'https://youtu.be/qgsbn8IoUqc?si=Xd_XsyAXLJrUokPR',
        'https://youtu.be/60LJ1HqNwKM?si=YIXakTyUZ30SexBN',
        'https://youtu.be/OSpbQc4DZCA?si=ARZu3LfL5mh0ttLv',
        'https://youtu.be/kDPNcxHGkmY?si=cx2KsBd2CqZvJn8T',
        'https://youtu.be/QUyFoyHT8s8?si=k9OcGU_08GmT-wSV',
        'https://youtu.be/Ays7xJK8UF8?si=Dw65IWhslYQ48tyu',
        'https://youtu.be/I81r9C5Mk14?si=ihHDsYJKexxidNG-',
        #Marathi Lavni
        "https://www.youtube.com/watch?v=mW67u_hWiSo&list=PLCwwkEaTIIWnQxuvnQCzKV5syKbLKwKc_&index=2",
        "https://www.youtube.com/watch?v=7R7QJkznJGU&list=PLCwwkEaTIIWnQxuvnQCzKV5syKbLKwKc_&index=3",
        "https://www.youtube.com/watch?v=E1aHfW80QAo&list=PLCwwkEaTIIWnQxuvnQCzKV5syKbLKwKc_&index=4",
        "https://www.youtube.com/watch?v=xFPTm9YwnwY&list=PLCwwkEaTIIWnQxuvnQCzKV5syKbLKwKc_&index=8",
        "https://www.youtube.com/watch?v=oEmD35XlsKU&list=PLCwwkEaTIIWnQxuvnQCzKV5syKbLKwKc_&index=11",
        "https://www.youtube.com/watch?v=7A3VW3Nws2Y&list=PLCwwkEaTIIWnQxuvnQCzKV5syKbLKwKc_&index=12",
        "https://www.youtube.com/watch?v=bhawaANaBa8&list=PLCwwkEaTIIWnQxuvnQCzKV5syKbLKwKc_&index=13",
        "https://www.youtube.com/watch?v=096lImuOnsM&list=PLCwwkEaTIIWnQxuvnQCzKV5syKbLKwKc_&index=14",
        "https://www.youtube.com/watch?v=pck81IGEq2E&list=PLCwwkEaTIIWnQxuvnQCzKV5syKbLKwKc_&index=22",
        "https://www.youtube.com/watch?v=Qf02VW7lDXE",
        'https://youtu.be/r6tU3GvJ5so?si=hrIpqnRApaJeea6R',
        'https://youtu.be/ThpuubLgN2A?si=qakunAzg8PMAM81B',
        'https://youtu.be/CukNxuj_MRk?si=6bhPCC3BIl-WBozR',
        'https://youtu.be/5g0RsS_KgvU?si=IHKbFgydFWxJnb0t',
        'https://youtu.be/aFGX888aZog?si=pc1aUO6pYAP3kc0Y',
        'https://youtu.be/7Qb6txP21Cg?si=QMvOWRghYn8JJfPh',
        'https://youtu.be/xODpqB8cuzo?si=OLslSacHJbgxJSIl',
        'https://youtu.be/GT9Hi8uN7Ao?si=cvqG9LkA7wJPE90o',
        'https://youtu.be/yqB5OTQDTdk?si=OQZ_Cm-bEU6rJPgI',
        'https://youtu.be/p924kpg5vmg?si=x-jIfJ6dlvZVTh3n',
        # National Anthem
        "https://www.youtube.com/watch?v=r3TtgYuaVFk",
        "https://www.youtube.com/watch?v=HtMF973tXIY&list=PLKSJChgq3i1wFg6bACYZcQtX7c1XGLVy9",
        "https://www.youtube.com/watch?v=G9fVGrqzPdg&list=PLKSJChgq3i1wFg6bACYZcQtX7c1XGLVy9&index=4",
        "https://www.youtube.com/watch?v=VqEbJBchHWQ&list=PLKSJChgq3i1wFg6bACYZcQtX7c1XGLVy9&index=5",
        "https://www.youtube.com/watch?v=daDutno1SUk&list=PLKSJChgq3i1wFg6bACYZcQtX7c1XGLVy9&index=8",
        "https://www.youtube.com/watch?v=q2hyS7XmOyw",
        'https://www.youtube.com/watch?v=xilrm7CHTgs',
        'https://www.youtube.com/watch?v=sHSAOKYVzwg',
        'https://www.youtube.com/watch?v=YAdSIxlHE5o',
        'https://www.youtube.com/watch?v=vsfH9TNjD1s',
        'https://youtu.be/R1pQSRQgrpk?si=C99RtqSGF8MkbWH9',
        'https://youtu.be/yO5uDLgUHcM?si=ZR7Y99Sx-aba-sk8',
        'https://youtu.be/AiluGfljpro?si=0BgN_54hqfhwm-w5',
        'https://youtu.be/re7ULG--YNI?si=sx0Y1Ps5nkE6vlFv',
        'https://youtu.be/anXcOOfS5oQ?si=12C4HeSk1GVpcpk4',
        'https://youtu.be/hw3xL1onw5c?si=ruXwUwHOLN45UYsb',
        'https://youtu.be/Llz3UoffcSY?si=tABYODDotJwSyt1s',
        'https://youtu.be/-l5T0UawmCY?si=Z-9j7Obx1wk1biZN',
        'https://youtu.be/q7c6zqn4L8g?si=-i6FnyZ9K1-h8zE5',
        'https://youtu.be/600yzHtfIVU?si=tLXcF3YEfe5fv6sB',
        'https://youtu.be/AZg3RY4VkhM?si=6yYztGMY37BqS7Nw',
        'https://youtu.be/mSt-Vnrlnx0?si=eWgl-vfyvWG9efea',
        # Marathi Bhavgeet
        'https://www.youtube.com/watch?v=V5xeOa8Ufv4&list=RDQMwhCcRL30wbI&start_radio=1',
        'https://www.youtube.com/watch?v=yGliZ8Fz4dc&list=RDQMwhCcRL30wbI&index=2',
        'https://www.youtube.com/watch?v=lvKFhbyMJm4&list=RDQMwhCcRL30wbI&index=5',
        'https://www.youtube.com/watch?v=Flv4Aq_Yxc8&list=RDQMwhCcRL30wbI&index=6',
        'https://www.youtube.com/watch?v=Pl41q9YH8LE&list=RDQMwhCcRL30wbI&index=8',
        'https://www.youtube.com/watch?v=nMhTwlnFius&list=RDQMwhCcRL30wbI&index=11',
        'https://www.youtube.com/watch?v=3bVbAD-Ws6E&list=RDQMwhCcRL30wbI&index=13',
        'https://www.youtube.com/watch?v=fZLaVGWfJwc&list=RDQMwhCcRL30wbI&index=32',
        'https://www.youtube.com/watch?v=OSOp2y1N5yA&list=RDQMwhCcRL30wbI&index=34',
        'https://www.youtube.com/watch?v=nXbQ-dItErg&list=RDQMwhCcRL30wbI&index=18',
        'https://youtu.be/mYvSHjO9LXA?si=ZI_hn40ggJb7TYzU',
        'https://youtu.be/oEPkPX3pI68?si=wEAvogxnAcTSe4jm',
        'https://youtu.be/M5ICKRpqeR4?si=xZ_0nI5tP8u6Nu0l',
        'https://youtu.be/VzDZtN5w8hA?si=7xG9w1BoUwc_3iNE',
        'https://youtu.be/29utf9DFxos?si=k8NcFmjA6-Swrdmh',
        'https://youtu.be/q4qSOpjP6Ig?si=uiixw9vwt0jBoa0d',
        'https://youtu.be/K0CXNZyGVeU?si=Dh6oYVywUDcGlz2F',
        'https://youtu.be/cm41EAWKddQ?si=ul6szYEJnSLwvyl0',
        'https://youtu.be/tFJfvoVw5i0?si=OYe_Xw4ZGgh0icmS',
        'https://youtu.be/1NM-kwpjc0M?si=rZA7_v9hm8hJ8svR',
        'https://youtu.be/R_q2ozcwcsI?si=J11nPVy6IWA_vdat',
        'https://youtu.be/B-aaz1tZuFs?si=OBSD5HjsVbEFVFDI',
        'https://youtu.be/o8vEdhKHMms?si=qEWhn0Am38fj5wIx'
        ]

labels = [0]*21 + [1]*20 + [2]*22 + [3]*20 + [4]*22 + [5]*23
# Create a list to store information about each processed song
songs_info = []

# Loop through the URLs and process each one
for idx, url in enumerate(urls, start=1):
    song_info = download_and_convert_to_mfcc(url, idx, labels[idx])
    if song_info:
        songs_info.append(song_info)

# Create a DataFrame from the songs information and save it as a CSV
songs_df = pd.DataFrame(songs_info)
summary_csv_path = os.path.join(output_folder, 'songs_summary.csv')
songs_df.to_csv(summary_csv_path, index=False)
print(f"Summary CSV saved as: {summary_csv_path}")
