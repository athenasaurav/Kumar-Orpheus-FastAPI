import requests
import argparse
from datetime import datetime
import os
import sys

def generate_speech(text, voice="tara", server_ip="34.150.166.116", port="5005"):
    """
    Generate speech from text using the Orpheus TTS server
    
    Args:
        text (str): The text to convert to speech
        voice (str): The voice to use (default: tara)
        server_ip (str): The server IP address
        port (str): The server port (default: 5005)
    
    Returns:
        str: Path to the saved audio file
    """
    # Create output directory if it doesn't exist
    os.makedirs("audio_outputs", exist_ok=True)
    
    # Generate unique filename using timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"audio_outputs/{voice}_{timestamp}.wav"
    
    # API endpoint
    url = f"http://{server_ip}:{port}/v1/audio/speech"
    
    # Request payload
    payload = {
        "input": text,
        "model": "orpheus",
        "voice": voice,
        "response_format": "wav",
        "speed": 1.0
    }
    
    try:
        print(f"Sending request to {url}")
        print(f"Using voice: {voice}")
        print(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")
        
        # Send request and stream the response to file
        response = requests.post(url, json=payload, stream=True)
        
        if response.status_code != 200:
            print(f"Error: Server returned status code {response.status_code}")
            print(f"Error message: {response.text}")
            return None
            
        # Save the audio file
        with open(output_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        print(f"\nSuccess! Audio saved to: {output_file}")
        return output_file
        
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to server: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Generate speech from text using Orpheus TTS")
    parser.add_argument("--text", type=str, help="Text to convert to speech")
    parser.add_argument("--voice", type=str, default="tara", 
                       help="Voice to use (tara, leah, jess, leo, dan, mia, zac, zoe)")
    parser.add_argument("--server", type=str, default="34.150.166.116",
                       help="Server IP address")
    parser.add_argument("--port", type=str, default="5005",
                       help="Server port")
    
    args = parser.parse_args()
    
    # If no text argument, try to read from stdin
    if not args.text:
        if not sys.stdin.isatty():
            args.text = sys.stdin.read().strip()
        else:
            args.text = input("Enter text to convert to speech: ")
    
    if not args.text:
        print("Error: No text provided")
        sys.exit(1)
    
    # Generate speech
    output_file = generate_speech(
        text=args.text,
        voice=args.voice,
        server_ip=args.server,
        port=args.port
    )
    
    if not output_file:
        sys.exit(1)

if __name__ == "__main__":
    main()
