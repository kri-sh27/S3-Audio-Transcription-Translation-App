import streamlit as st
import boto3
import openai
import os
from dotenv import load_dotenv
import tempfile

# Load environment variables
load_dotenv()

# Configure AWS credentials
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_region = os.getenv('AWS_REGION')
bucket_name = os.getenv('S3_BUCKET_NAME')

# Configure OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_region
)

def list_s3_audio_files():
    """List audio files in S3 bucket"""
    response = s3_client.list_objects_v2(Bucket=bucket_name)
    audio_files = []
    for obj in response.get('Contents', []):
        if obj['Key'].lower().endswith(('.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.wav', '.webm')):
            audio_files.append(obj['Key'])
    return audio_files

def transcribe_audio(file_path):
    """Transcribe audio file using OpenAI Whisper API"""
    with open(file_path, "rb") as audio_file:
        transcript = openai.Audio.transcribe(
            file=audio_file,
            model="whisper-1",
            response_format="text"
        )
    return transcript

def translate_text(text, target_language):
    """Translate text using OpenAI GPT"""
    prompt = f"Translate the following English text to {target_language}:\n\n{text}"
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a professional translator."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# Streamlit UI
st.title("🎙️ S3 Audio Transcription & Translation App")

# Sidebar with file selection
st.sidebar.header("Select Audio File")
audio_files = list_s3_audio_files()
selected_file = st.sidebar.selectbox("Choose an audio file", audio_files)

# Language selection
languages = {
    "Original (No Translation)": None,
    "Hindi": "Hindi",
    "Marathi": "Marathi",
    "Japanese": "Japanese",
    "Spanish": "Spanish",
    "French": "French",
    "German": "German"
}
target_language = st.sidebar.selectbox("Select Translation Language", list(languages.keys()))

if selected_file:
    st.write(f"Selected file: **{selected_file}**")
    
    # Add a transcribe button
    if st.button("Transcribe & Translate Audio"):
        with st.spinner("Downloading and transcribing audio..."):
            temp_file_path = None
            try:
                # Create a temporary file
                temp_file_path = os.path.join(tempfile.gettempdir(), f"temp_audio_{os.getpid()}{os.path.splitext(selected_file)[1]}")
                
                # Download file from S3
                s3_client.download_file(bucket_name, selected_file, temp_file_path)
                
                # Transcribe the audio
                transcript = transcribe_audio(temp_file_path)
                
                # Display the original transcript
                st.success("Transcription completed!")
                st.markdown("### Original Transcript:")
                st.write(transcript)
                
                # Translate if a language is selected
                if languages[target_language]:
                    with st.spinner(f"Translating to {target_language}..."):
                        translated_text = translate_text(transcript, languages[target_language])
                        st.markdown(f"### {target_language} Translation:")
                        st.write(translated_text)
                        
                        # Option to download translated transcript
                        col1, col2 = st.columns(2)
                        with col1:
                            st.download_button(
                                label="Download Original Transcript",
                                data=transcript,
                                file_name=f"{selected_file}_transcript_original.txt",
                                mime="text/plain"
                            )
                        with col2:
                            st.download_button(
                                label=f"Download {target_language} Translation",
                                data=translated_text,
                                file_name=f"{selected_file}_transcript_{target_language.lower()}.txt",
                                mime="text/plain"
                            )
                else:
                    # Option to download original transcript only
                    st.download_button(
                        label="Download Transcript",
                        data=transcript,
                        file_name=f"{selected_file}_transcript.txt",
                        mime="text/plain"
                    )
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
            
            finally:
                # Clean up temporary file
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.close(os.open(temp_file_path, os.O_RDONLY))
                        os.remove(temp_file_path)
                    except Exception as e:
                        st.warning(f"Could not remove temporary file: {str(e)}")

# Add some usage instructions
with st.sidebar.expander("ℹ️ How to use"):
    st.write("""
    1. Select an audio file from the dropdown menu
    2. Choose your desired translation language
    3. Click the 'Transcribe & Translate Audio' button
    4. Wait for the transcription and translation to complete
    5. View and download the results
    """)

# Add footer
st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ using OpenAI Whisper By Krishnat ❤️❤️❤️")