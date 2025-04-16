            if audio is None:
                print("Received end-of-stream marker")
                break
            
            # Store the audio segment for return value
            audio_segments.append(audio)
            
            # Write to file if needed
            if wav_file:
                write_buffer.extend(audio)
                
                # Flush buffer if it's large enough
                if len(write_buffer) >= buffer_max_size:
                    wav_file.writeframes(write_buffer)
                    write_buffer = bytearray()  # Reset buffer
        
        except queue.Empty:
            # No data available right now
            current_time = time.time()
            
            # Periodically check if producer is done
            if current_time - last_check_time > check_interval:
                last_check_time = current_time
                
                # If producer is done and queue is empty, we're finished
                if producer_done_event.is_set() and audio_queue.empty():
                    print("Producer done and queue empty - finishing consumer")
                    break
                
                # Flush buffer periodically even if not full
                if wav_file and len(write_buffer) > 0:
                    wav_file.writeframes(write_buffer)
                    write_buffer = bytearray()  # Reset buffer
    
    # Extra safety check - ensure thread is done
    if thread.is_alive():
        print("Waiting for token processor thread to complete...")
        thread.join(timeout=10.0)
        if thread.is_alive():
            print("WARNING: Token processor thread did not complete within timeout")
    
    # Final flush of any remaining data
    if wav_file and len(write_buffer) > 0:
        print(f"Final buffer flush: {len(write_buffer)} bytes")
        wav_file.writeframes(write_buffer)
    
    # Close WAV file if opened
    if wav_file:
        wav_file.close()
        if output_file:
            print(f"Audio saved to {output_file}")
    
    # Calculate and print detailed performance metrics
    if audio_segments:
        total_bytes = sum(len(segment) for segment in audio_segments)
        duration = total_bytes / (2 * SAMPLE_RATE)  # 2 bytes per sample at 24kHz
        total_time = time.time() - perf_monitor.start_time
        realtime_factor = duration / total_time if total_time > 0 else 0
        
        print(f"Generated {len(audio_segments)} audio segments")
        print(f"Generated {duration:.2f} seconds of audio in {total_time:.2f} seconds")
        print(f"Realtime factor: {realtime_factor:.2f}x")
        
        if realtime_factor < 1.0:
            print("⚠️ Warning: Generation is slower than realtime")
        else:
            print(f"✓ Generation is {realtime_factor:.1f}x faster than realtime")
    
    return audio_segments

def stream_audio(audio_buffer):
    """Stream audio buffer to output device with error handling."""
    if audio_buffer is None or len(audio_buffer) == 0:
        return
    
    try:
        # Convert bytes to NumPy array (16-bit PCM)
        audio_data = np.frombuffer(audio_buffer, dtype=np.int16)
        
        # Normalize to float in range [-1, 1] for playback
        audio_float = audio_data.astype(np.float32) / 32767.0
        
        # Play the audio with proper device selection and error handling
        sd.play(audio_float, SAMPLE_RATE)
        sd.wait()
    except Exception as e:
        print(f"Audio playback error: {e}")

import re
import numpy as np
from io import BytesIO
import wave

def split_text_into_sentences(text):
    """Split text into sentences with a more reliable approach."""
    # We'll use a simple approach that doesn't rely on variable-width lookbehinds
    # which aren't supported in Python's regex engine
    
    # First, split on common sentence ending punctuation
    # This isn't perfect but works for most cases and avoids the regex error
    parts = []
    current_sentence = ""
    
    for char in text:
        current_sentence += char
        
        # If we hit a sentence ending followed by a space, consider this a potential sentence end
        if char in (' ', '\n', '\t') and len(current_sentence) > 1:
            prev_char = current_sentence[-2]
            if prev_char in ('.', '!', '?'):
                # Check if this is likely a real sentence end and not an abbreviation
                # (Simple heuristic: if there's a space before the period, it's likely a real sentence end)
                if len(current_sentence) > 3 and current_sentence[-3] not in ('.', ' '):
                    parts.append(current_sentence.strip())
                    current_sentence = ""
    
    # Add any remaining text
    if current_sentence.strip():
        parts.append(current_sentence.strip())
    
    # Combine very short segments to avoid tiny audio files
    min_chars = 20  # Minimum reasonable sentence length
    combined_sentences = []
    i = 0
    
    while i < len(parts):
        current = parts[i]
        
        # If this is a short sentence and not the last one, combine with next
        while i < len(parts) - 1 and len(current) < min_chars:
            i += 1
            current += " " + parts[i]
            
        combined_sentences.append(current)
        i += 1
    
    return combined_sentences

def generate_speech_from_api(prompt, voice=DEFAULT_VOICE, output_file=None, temperature=TEMPERATURE, 
                     top_p=TOP_P, max_tokens=MAX_TOKENS, repetition_penalty=None, 
                     use_batching=True, max_batch_chars=1000):
    """Generate speech from text using Orpheus model with performance optimizations."""
    print(f"Starting speech generation for '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'") 
    print(f"Using voice: {voice}, GPU acceleration: {'Yes (High-end)' if HIGH_END_GPU else 'Yes' if torch.cuda.is_available() else 'No'}")
    
    # Reset performance monitor
    global perf_monitor
    perf_monitor = PerformanceMonitor()
    
    start_time = time.time()
    
    # For shorter text, use the standard non-batched approach
    if not use_batching or len(prompt) < max_batch_chars:
        # Note: we ignore any provided repetition_penalty and always use the hardcoded value
        # This ensures consistent quality regardless of what might be passed in
        result = tokens_decoder_sync(
            generate_tokens_from_api(
                prompt=prompt, 
                voice=voice,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                repetition_penalty=REPETITION_PENALTY  # Always use hardcoded value
            ),
            output_file=output_file
        )
        
        # Report final performance metrics
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total speech generation completed in {total_time:.2f} seconds")
        
        return result
    
    # For longer text, use sentence-based batching
    print(f"Using sentence-based batching for text with {len(prompt)} characters")
    
    # Split the text into sentences
    sentences = split_text_into_sentences(prompt)
    print(f"Split text into {len(sentences)} segments")
    
    # Create batches by combining sentences up to max_batch_chars
    batches = []
    current_batch = ""
    
    for sentence in sentences:
        # If adding this sentence would exceed the batch size, start a new batch
        if len(current_batch) + len(sentence) > max_batch_chars and current_batch:
            batches.append(current_batch)
            current_batch = sentence
        else:
            # Add separator space if needed
            if current_batch:
                current_batch += " "
            current_batch += sentence
    
    # Add the last batch if it's not empty
    if current_batch:
        batches.append(current_batch)
    
    print(f"Created {len(batches)} batches for processing")
    
    # Process each batch and collect audio segments
    all_audio_segments = []
    batch_temp_files = []
    
    for i, batch in enumerate(batches):
        print(f"Processing batch {i+1}/{len(batches)} ({len(batch)} characters)")
        
        # Create a temporary file for this batch if an output file is requested
        temp_output_file = None
        if output_file:
            temp_output_file = f"outputs/temp_batch_{i}_{int(time.time())}.wav"
            batch_temp_files.append(temp_output_file)
        
        # Generate speech for this batch
        batch_segments = tokens_decoder_sync(
            generate_tokens_from_api(
                prompt=batch,
                voice=voice,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                repetition_penalty=REPETITION_PENALTY
            ),
            output_file=temp_output_file
        )
        
        # Add to our collection
        all_audio_segments.extend(batch_segments)
    
    # If an output file was requested, stitch together the temporary files
    if output_file and batch_temp_files:
        # Stitch together WAV files
        stitch_wav_files(batch_temp_files, output_file)
        
        # Clean up temporary files
        for temp_file in batch_temp_files:
            try:
                os.remove(temp_file)
            except Exception as e:
                print(f"Warning: Could not remove temporary file {temp_file}: {e}")
    
    # Report final performance metrics
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate combined duration
    if all_audio_segments:
        total_bytes = sum(len(segment) for segment in all_audio_segments)