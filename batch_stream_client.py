#!/usr/bin/env python3
"""
Orpheus TTS Batch Stream Client
Processes multiple text lines concurrently from a file, saves audio,
and provides detailed logging including the source text per chunk.
"""

import argparse
import asyncio
import aiohttp
import aiofiles
import os
import sys
import time
import json
import wave
import io
import logging
from datetime import datetime
from typing import List, Tuple

# Set up logging
def setup_logging(log_file: str = "batch_tts.log"):
    """Configure logging to both file and console"""
    # Create formatter with a simpler datetime format
    formatter = logging.Formatter(
        '%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'  # Removed problematic microseconds
    )

    # Set up file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Set up console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

async def process_line(session: aiohttp.ClientSession, line_data: Tuple[str, str], server_url: str, save_path: str, line_index: int):
    """Processes a single line (filename, text) by sending a request and handling the stream."""
    
    output_filename, text_to_speak = line_data
    output_filepath = os.path.join(save_path, output_filename)
    logger = logging.getLogger()
    
    # Prepare request payload
    payload = {
        "input": text_to_speak,
        "model": "orpheus",
        "voice": "tara",
        "response_format": "wav",
        "speed": 1.0
    }
    
    request_start_time = time.time()
    logger.info(f"Req {line_index+1:02d}: Starting request for '{output_filename}'...")
    
    first_byte_received = False
    chunk_count = 0
    total_bytes = 0
    audio_chunks = []  # Store chunks in memory temporarily
    
    try:
        async with session.post(server_url, json=payload) as response:
            request_end_time = time.time()
            logger.info(f"Req {line_index+1:02d}: Sent request ({request_end_time - request_start_time:.3f}s) for '{output_filename}'. Status: {response.status}")

            if response.status != 200:
                error_text = await response.text()
                logger.error(f"Req {line_index+1:02d}: ERROR for '{output_filename}': {response.status} - {error_text[:200]}")
                return

            last_chunk_time = request_end_time
            
            async for chunk in response.content.iter_chunked(4096):
                if chunk:
                    current_time = time.time()
                    chunk_size = len(chunk)
                    total_bytes += chunk_size
                    chunk_count += 1
                    
                    if not first_byte_received:
                        first_byte_time = current_time
                        latency = first_byte_time - request_end_time
                        logger.info(f"Req {line_index+1:02d}: First audio byte for '{output_filename}' (latency: {latency:.3f}s)")
                        first_byte_received = True

                    chunk_interval = current_time - last_chunk_time
                    chunk_rate = chunk_size / chunk_interval if chunk_interval > 0 else 0
                    
                    logger.info(
                        f"Req {line_index+1:02d}: "
                        f"Chunk {chunk_count:3d} received: {chunk_size:6,d} bytes | "
                        f"Interval: {chunk_interval*1000:6.1f}ms | "
                        f"Rate: {chunk_rate/1024:6.1f} KB/s | "
                        f"FOR_TTS: {text_to_speak[:80]}{'...' if len(text_to_speak) > 80 else ''}"
                    )
                    
                    audio_chunks.append(chunk)
                    last_chunk_time = current_time
            
            # After receiving all chunks, write to WAV file with proper headers
            os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
            with wave.open(output_filepath, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(24000)  # 24kHz
                for chunk in audio_chunks:
                    wav_file.writeframes(chunk)
            
            stream_end_time = time.time()
            total_req_duration = stream_end_time - request_start_time
            streaming_duration = stream_end_time - first_byte_time if first_byte_received else 0
            avg_streaming_rate = (total_bytes / 1024) / streaming_duration if streaming_duration > 0 else 0
            
            logger.info(f"Req {line_index+1:02d}: Finished '{output_filename}'. Total time: {total_req_duration:.3f}s, Chunks: {chunk_count}, Bytes: {total_bytes:,d}, Avg Rate: {avg_streaming_rate:.1f} KB/s")

    except aiohttp.ClientError as e:
        logger.error(f"Req {line_index+1:02d}: Connection Error for '{output_filename}': {e}")
    except Exception as e:
        logger.error(f"Req {line_index+1:02d}: Unexpected Error for '{output_filename}': {e}")
        import traceback
        logger.error(traceback.format_exc())

async def main(data_file: str, num_lines: int, server_ip: str, port: str, save_path: str, concurrency_limit: int, log_file: str):
    """Main function to read data, create tasks, and run them concurrently."""
    
    # Setup logging
    logger = setup_logging(log_file)
    
    server_url = f"http://{server_ip}:{port}/v1/audio/speech"
    lines_to_process: List[Tuple[str, str]] = []

    logger.info(f"Reading first {num_lines} lines from '{data_file}'...")
    try:
        async with aiofiles.open(data_file, 'r', encoding='utf-8') as f:
            count = 0
            async for line in f:
                if count >= num_lines:
                    break
                line = line.strip()
                if not line or '|' not in line:
                    continue
                
                parts = line.split('|', 1)
                if len(parts) == 2:
                    filename, text = parts[0].strip(), parts[1].strip()
                    if filename.endswith(".wav"):
                         lines_to_process.append((filename, text))
                         count += 1
                    else:
                         logger.warning(f"Skipping line {count+1}: Filename '{filename}' does not end with .wav")
                else:
                    logger.warning(f"Skipping malformed line {count+1}: {line}")
                    
    except FileNotFoundError:
        logger.error(f"Error: Data file '{data_file}' not found.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error reading data file '{data_file}': {e}")
        sys.exit(1)

    if not lines_to_process:
        logger.error("No valid lines found to process.")
        sys.exit(0)
        
    logger.info(f"Found {len(lines_to_process)} lines to process. Starting concurrent requests (limit: {concurrency_limit})...")
    
    # Use a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(concurrency_limit)
    
    async def safe_process_line(session, line_data, server_url, save_path, index):
        async with semaphore:
            await process_line(session, line_data, server_url, save_path, index)

    start_time = time.time()
    
    # Use a single session for connection pooling
    async with aiohttp.ClientSession() as session:
        tasks = [
            safe_process_line(session, line_data, server_url, save_path, i)
            for i, line_data in enumerate(lines_to_process)
        ]
        await asyncio.gather(*tasks)

    end_time = time.time()
    logger.info(f"Batch processing completed in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orpheus TTS Batch Stream Client")
    parser.add_argument(
        "--data-file", 
        type=str, 
        default="data.txt",
        help="Path to the data file (format: filename.wav|text)"
    )
    parser.add_argument(
        "--num-lines", 
        type=int, 
        default=16,
        help="Number of lines from the data file to process"
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
        "--save-path", 
        type=str, 
        default="batch_outputs",
        help="Directory to save the generated audio files"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Maximum number of concurrent requests"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="batch_tts.log",
        help="Log file path"
    )
    
    args = parser.parse_args()
    
    # Ensure save path exists
    os.makedirs(args.save_path, exist_ok=True)
    
    asyncio.run(main(
        data_file=args.data_file,
        num_lines=args.num_lines,
        server_ip=args.server,
        port=args.port,
        save_path=args.save_path,
        concurrency_limit=args.concurrency,
        log_file=args.log_file
    )) 