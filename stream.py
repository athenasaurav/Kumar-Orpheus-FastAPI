import requests
import argparse
import sounddevice as sd
import numpy as np
from datetime import datetime
import os
import sys
import wave
import threading
import queue
import time

# Global audio queue for real-time playback
audio_queue = queue.Queue()
is_playing = False

def play_audio_stream():
    """
    Continuously play audio chunks from the queue in real-time
    """
    global is_playing
    
    # Audio parameters
    sample_rate = 24000  # Orpheus uses 24kHz
    channels = 1  # Mono audio
    
    # Start audio stream
    with sd.OutputStream(samplerate=sample_rate, channels=channels, dtype=np.int16) as stream:
        is_playing = True
        
        while is_playing:
            try:
                # Get audio chunk from queue with timeout
                audio_chunk = audio_queue.get(timeout=0.1)
                
                if audio_chunk is None:
                    # None is our sentinel value to stop playback
                    break
                    
                # Convert bytes to numpy array
                audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
                
                # Write to audio stream
                stream.write(audio_data)
                
            except queue.Empty:
                # No audio data available, continue waiting
                continue
            except Exception as e:
                print(f"Error in audio playback: {e}")
                break
    
    is_playing = False

def generate_speech(text, voice="tara", server_ip="34.150.166.116", port="5005", save_audio=True, play_audio=True):
    """
    Generate speech from text using the Orpheus TTS server with real-time streaming
    
    Args:
        text (str): The text to convert to speech
        voice (str): The voice to use (default: tara)
        server_ip (str): The server IP address
        port (str): The server port (default: 5005)
        save_audio (bool): Whether to save the audio to a file
        play_audio (bool): Whether to play the audio in real-time
    
    Returns:
        str: Path to the saved audio file (if save_audio=True)
    """
    global is_playing
    output_file = None
    wav_file = None
    
    # Create output directory if saving audio
    if save_audio:
        os.makedirs("audio_outputs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"audio_outputs/{voice}_{timestamp}.wav"
        
        # Prepare WAV file
        wav_file = wave.open(output_file, 'wb')
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(24000)  # 24kHz
    
    # Start audio playback thread if requested
    if play_audio:
        playback_thread = threading.Thread(target=play_audio_stream)
        playback_thread.start()
    
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
        
        # Send request and stream the response
        response = requests.post(url, json=payload, stream=True)
        
        if response.status_code != 200:
            print(f"Error: Server returned status code {response.status_code}")
            print(f"Error message: {response.text}")
            return None
        
        # Process the audio stream
        for chunk in response.iter_content(chunk_size=4096):
            if chunk:
                # Save to file if requested
                if save_audio and wav_file:
                    wav_file.writeframes(chunk)
                
                # Add to playback queue if playing
                if play_audio:
                    audio_queue.put(chunk)
        
        # Signal end of audio stream
        if play_audio:
            audio_queue.put(None)
            
        # Close WAV file if open
        if wav_file:
            wav_file.close()
            print(f"\nSuccess! Audio saved to: {output_file}")
        
        # Wait for playback to finish if playing
        if play_audio:
            playback_thread.join()
        
        return output_file
        
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to server: {e}")
        return None
    finally:
        # Cleanup
        if wav_file:
            wav_file.close()
        if play_audio:
            is_playing = False
            audio_queue.put(None)  # Ensure playback thread exits

def main():
    parser = argparse.ArgumentParser(description="Generate speech from text using Orpheus TTS")
    parser.add_argument("--text", type=str, help="Text to convert to speech")
    parser.add_argument("--voice", type=str, default="tara", 
                       help="Voice to use (tara, leah, jess, leo, dan, mia, zac, zoe)")
    parser.add_argument("--server", type=str, default="34.150.166.116",
                       help="Server IP address")
    parser.add_argument("--port", type=str, default="5005",
                       help="Server port")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save the audio to a file")
    parser.add_argument("--no-play", action="store_true",
                       help="Don't play the audio in real-time")
    
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
    
    try:
        # Generate speech
        output_file = generate_speech(
            text=args.text,
            voice=args.voice,
            server_ip=args.server,
            port=args.port,
            save_audio=not args.no_save,
            play_audio=not args.no_play
        )
        
        if not output_file and not args.no_save:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)

if __name__ == "__main__":
    main()