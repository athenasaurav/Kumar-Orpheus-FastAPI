#!/usr/bin/env python3
"""
Orpheus TTS Stream Client - Server Version
Optimized for server-side usage with detailed metrics and logging
"""

import requests
import argparse
import numpy as np
from datetime import datetime
import os
import sys
import wave
import time
import json

class OrpheusStreamClient:
    def __init__(self, server_ip="0.0.0.0", port="5005", voice="tara", save_path="audio_outputs"):
        self.server_ip = server_ip
        self.port = port
        self.voice = voice
        self.save_path = save_path
        self.url = f"http://{server_ip}:{port}/v1/audio/speech"
        
        # Ensure save directory exists if needed
        if save_path:
            os.makedirs(save_path, exist_ok=True)
    
    def _get_timestamp(self):
        """Get current timestamp with millisecond precision"""
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    
    def _log(self, message, with_timestamp=True):
        """Log message with optional timestamp"""
        if with_timestamp:
            print(f"[{self._get_timestamp()}] {message}")
        else:
            print(message)
    
    def generate_speech(self, text, save_audio=False):
        """
        Generate speech from text with detailed metrics
        
        Args:
            text (str): Text to convert to speech
            save_audio (bool): Whether to save the audio to a file
        
        Returns:
            dict: Performance metrics and file path if saved
        """
        metrics = {
            'start_time': time.time(),
            'first_byte_time': None,
            'end_time': None,
            'chunks': [],
            'total_bytes': 0,
            'chunk_count': 0,
            'output_file': None
        }
        
        # Setup WAV file if saving
        wav_file = None
        if save_audio:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"{self.save_path}/{self.voice}_{timestamp}.wav"
            metrics['output_file'] = output_path
            wav_file = wave.open(output_path, 'wb')
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(24000)  # 24kHz
        
        try:
            # Log start of request
            self._log("\nStarting TTS request")
            self._log(f"Voice: {self.voice}", False)
            self._log(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}", False)
            self._log(f"Server: {self.url}", False)
            
            # Prepare request
            payload = {
                "input": text,
                "model": "orpheus",
                "voice": self.voice,
                "response_format": "wav",
                "speed": 1.0
            }
            
            # Send request and get response
            request_time = time.time()
            self._log("Sending request...")
            
            response = requests.post(self.url, json=payload, stream=True)
            
            if response.status_code != 200:
                self._log(f"Error: Server returned status code {response.status_code}")
                if response.text:
                    self._log(f"Error details: {response.text}")
                return None
                
            self._log("Request accepted by server")
            
            # Process the streaming response
            last_chunk_time = request_time
            
            for chunk in response.iter_content(chunk_size=4096):
                if chunk:
                    current_time = time.time()
                    chunk_size = len(chunk)
                    metrics['total_bytes'] += chunk_size
                    metrics['chunk_count'] += 1
                    
                    # Record first byte timing
                    if metrics['first_byte_time'] is None:
                        metrics['first_byte_time'] = current_time
                        latency = current_time - request_time
                        self._log(f"First audio byte received (latency: {latency:.3f}s)")
                    
                    # Calculate chunk metrics
                    chunk_interval = current_time - last_chunk_time
                    chunk_rate = chunk_size / chunk_interval if chunk_interval > 0 else 0
                    
                    # Store chunk metrics
                    chunk_metrics = {
                        'timestamp': current_time,
                        'size': chunk_size,
                        'interval': chunk_interval,
                        'rate': chunk_rate
                    }
                    metrics['chunks'].append(chunk_metrics)
                    
                    # Log chunk details
                    self._log(
                        f"Chunk {metrics['chunk_count']:3d} received: "
                        f"{chunk_size:6,d} bytes | "
                        f"Interval: {chunk_interval*1000:6.1f}ms | "
                        f"Rate: {chunk_rate/1024:6.1f} KB/s"
                    )
                    
                    # Save audio if requested
                    if wav_file:
                        wav_file.writeframes(chunk)
                    
                    last_chunk_time = current_time
            
            # Record end time and calculate final metrics
            metrics['end_time'] = time.time()
            
            # Print final statistics
            self._log("\nStream completed")
            self._print_final_stats(metrics, request_time)
            
            return metrics
            
        except requests.exceptions.RequestException as e:
            self._log(f"Error connecting to server: {e}")
            return None
            
        finally:
            if wav_file:
                wav_file.close()
                if save_audio and metrics['output_file']:
                    self._log(f"Audio saved to: {metrics['output_file']}")
    
    def _print_final_stats(self, metrics, request_time):
        """Print final statistics from the metrics"""
        total_duration = metrics['end_time'] - metrics['start_time']
        
        print(f"\nFinal Statistics:")
        print(f"{'='*50}")
        print(f"Total duration: {total_duration:.3f}s")
        print(f"Total chunks: {metrics['chunk_count']}")
        print(f"Total bytes: {metrics['total_bytes']:,d}")
        
        if metrics['chunk_count'] > 0:
            print(f"Average chunk size: {metrics['total_bytes']/metrics['chunk_count']:,.1f} bytes")
        
        if metrics['first_byte_time']:
            latency = metrics['first_byte_time'] - request_time
            streaming_duration = metrics['end_time'] - metrics['first_byte_time']
            streaming_rate = (metrics['total_bytes']/1024)/streaming_duration
            
            print(f"Time to first byte: {latency:.3f}s")
            print(f"Streaming duration: {streaming_duration:.3f}s")
            print(f"Average streaming rate: {streaming_rate:.1f} KB/s")
        
        if metrics['chunks']:
            rates = [chunk['rate']/1024 for chunk in metrics['chunks']]  # Convert to KB/s
            print(f"Min chunk rate: {min(rates):.1f} KB/s")
            print(f"Max chunk rate: {max(rates):.1f} KB/s")
            print(f"Avg chunk rate: {sum(rates)/len(rates):.1f} KB/s")
        
        print(f"{'='*50}")

def main():
    parser = argparse.ArgumentParser(
        description="Orpheus TTS Stream Client - Server Version"
    )
    parser.add_argument(
        "--text", 
        type=str, 
        help="Text to convert to speech"
    )
    parser.add_argument(
        "--voice", 
        type=str, 
        default="tara",
        help="Voice to use (default: tara)"
    )
    parser.add_argument(
        "--server", 
        type=str, 
        default="0.0.0.0",
        help="Server IP address"
    )
    parser.add_argument(
        "--port", 
        type=str, 
        default="5005",
        help="Server port"
    )
    parser.add_argument(
        "--save", 
        action="store_true",
        help="Save the audio to a file"
    )
    parser.add_argument(
        "--metrics-file",
        type=str,
        help="Save detailed metrics to a JSON file"
    )
    
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
    
    # Create client and generate speech
    client = OrpheusStreamClient(
        server_ip=args.server,
        port=args.port,
        voice=args.voice
    )
    
    try:
        metrics = client.generate_speech(
            text=args.text,
            save_audio=args.save
        )
        
        # Save metrics if requested
        if args.metrics_file and metrics:
            # Convert timestamps to strings for JSON serialization
            metrics_json = {
                'start_time': datetime.fromtimestamp(metrics['start_time']).isoformat(),
                'end_time': datetime.fromtimestamp(metrics['end_time']).isoformat(),
                'first_byte_time': datetime.fromtimestamp(metrics['first_byte_time']).isoformat() if metrics['first_byte_time'] else None,
                'total_bytes': metrics['total_bytes'],
                'chunk_count': metrics['chunk_count'],
                'output_file': metrics['output_file'],
                'chunks': [
                    {
                        'timestamp': datetime.fromtimestamp(c['timestamp']).isoformat(),
                        'size': c['size'],
                        'interval': c['interval'],
                        'rate': c['rate']
                    }
                    for c in metrics['chunks']
                ]
            }
            
            with open(args.metrics_file, 'w') as f:
                json.dump(metrics_json, f, indent=2)
            print(f"\nDetailed metrics saved to: {args.metrics_file}")
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)

if __name__ == "__main__":
    main() 