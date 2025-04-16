            )
            
            if response.status_code != 200:
                print(f"Error: API request failed with status code {response.status_code}")
                print(f"Error details: {response.text}")
                # Retry on server errors (5xx) but not on client errors (4xx)
                if response.status_code >= 500:
                    retry_count += 1
                    wait_time = 2 ** retry_count  # Exponential backoff
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                return
            
            # Process the streamed response with better buffering
            buffer = ""
            token_counter = 0
            
            # Iterate through the response to get tokens
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]  # Remove the 'data: ' prefix
                        
                        if data_str.strip() == '[DONE]':
                            break
                            
                        try:
                            data = json.loads(data_str)
                            if 'choices' in data and len(data['choices']) > 0:
                                token_chunk = data['choices'][0].get('text', '')
                                for token_text in token_chunk.split('>'):
                                    token_text = f'{token_text}>'
                                    token_counter += 1
                                    perf_monitor.add_tokens()

                                    if token_text:
                                        yield token_text
                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON: {e}")
                            continue
            
            # Generation completed successfully
            generation_time = time.time() - start_time
            tokens_per_second = token_counter / generation_time if generation_time > 0 else 0
            print(f"Token generation complete: {token_counter} tokens in {generation_time:.2f}s ({tokens_per_second:.1f} tokens/sec)")
            return
            
        except requests.exceptions.Timeout:
            print(f"Request timed out after {REQUEST_TIMEOUT} seconds")
            retry_count += 1
            if retry_count < max_retries:
                wait_time = 2 ** retry_count
                print(f"Retrying in {wait_time} seconds... (attempt {retry_count+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print("Max retries reached. Token generation failed.")
                return
                
        except requests.exceptions.ConnectionError:
            print(f"Connection error to API at {API_URL}")
            retry_count += 1
            if retry_count < max_retries:
                wait_time = 2 ** retry_count
                print(f"Retrying in {wait_time} seconds... (attempt {retry_count+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print("Max retries reached. Token generation failed.")
                return

# The turn_token_into_id function is now imported from speechpipe.py
# This eliminates duplicate code and ensures consistent behavior

def convert_to_audio(multiframe: List[int], count: int) -> Optional[bytes]:
    """Convert token frames to audio with performance monitoring."""
    # Import here to avoid circular imports
    from .speechpipe import convert_to_audio as orpheus_convert_to_audio
    start_time = time.time()
    result = orpheus_convert_to_audio(multiframe, count)
    
    if result is not None:
        perf_monitor.add_audio_chunk()
        
    return result

async def tokens_decoder(token_gen) -> Generator[bytes, None, None]:
    """Simplified token decoder with early first-chunk processing for lower latency."""
    buffer = []
    count = 0
    
    # Use different thresholds for first chunk vs. subsequent chunks
    first_chunk_processed = False
    min_frames_first = 7  # Process after just 7 tokens for first chunk (ultra-low latency)
    min_frames_subsequent = 28  # Default for reliability after first chunk (4 chunks of 7)
    process_every = 7  # Process every 7 tokens (standard for Orpheus model)
    
    start_time = time.time()
    last_log_time = start_time
    token_count = 0
    
    async for token_text in token_gen:
        token = turn_token_into_id(token_text, count)
        if token is not None and token > 0:
            # Add to buffer using simple append (reliable method)
            buffer.append(token)
            count += 1
            token_count += 1
            
            # Log throughput periodically
            current_time = time.time()
            if current_time - last_log_time > 5.0:  # Every 5 seconds
                elapsed = current_time - start_time
                if elapsed > 0:
                    print(f"Token processing rate: {token_count/elapsed:.1f} tokens/second")
                last_log_time = current_time
            
            # Different processing paths based on whether first chunk has been processed
            if not first_chunk_processed:
                # For first audio output, process as soon as we have enough tokens for one chunk
                if count >= min_frames_first:
                    buffer_to_proc = buffer[-min_frames_first:]
                    
                    # Process the first chunk for immediate audio feedback
                    print(f"Processing first audio chunk with {len(buffer_to_proc)} tokens")
                    audio_samples = convert_to_audio(buffer_to_proc, count)
                    if audio_samples is not None:
                        first_chunk_processed = True  # Mark first chunk as processed
                        yield audio_samples
            else:
                # For subsequent chunks, use standard processing with larger batch
                if count % process_every == 0 and count >= min_frames_subsequent:
                    # Use simple slice operation - reliable and correct
                    buffer_to_proc = buffer[-min_frames_subsequent:]
                    
                    # Debug output to help diagnose issues
                    if count % 28 == 0:
                        print(f"Processing buffer with {len(buffer_to_proc)} tokens, total collected: {len(buffer)}")
                    
                    # Process the tokens
                    audio_samples = convert_to_audio(buffer_to_proc, count)
                    if audio_samples is not None:
                        yield audio_samples

def tokens_decoder_sync(syn_token_gen, output_file=None):
    """Optimized synchronous wrapper with parallel processing and efficient file I/O."""
    # Use a larger queue for high-end systems
    queue_size = 100 if HIGH_END_GPU else 50
    audio_queue = queue.Queue(maxsize=queue_size)
    audio_segments = []
    
    # If output_file is provided, prepare WAV file with buffered I/O
    wav_file = None
    if output_file:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        wav_file = wave.open(output_file, "wb")
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
    
    # Batch processing of tokens for improved throughput
    batch_size = 32 if HIGH_END_GPU else 16
    
    # Thread synchronization for proper completion detection
    producer_done_event = threading.Event()
    producer_started_event = threading.Event()
    
    # Convert the synchronous token generator into an async generator with batching
    async def async_token_gen():
        batch = []
        for token in syn_token_gen:
            batch.append(token)
            if len(batch) >= batch_size:
                for t in batch:
                    yield t
                batch = []
        # Process any remaining tokens in the final batch
        for t in batch:
            yield t

    async def async_producer():
        # Track performance with more granular metrics
        start_time = time.time()
        chunk_count = 0
        last_log_time = start_time
        
        try:
            # Signal that producer has started processing
            producer_started_event.set()
            
            async for audio_chunk in tokens_decoder(async_token_gen()):
                # Process each audio chunk from the decoder
                if audio_chunk:
                    audio_queue.put(audio_chunk)
                    chunk_count += 1
                    
                    # Log performance periodically
                    current_time = time.time()
                    if current_time - last_log_time >= 3.0:  # Every 3 seconds
                        elapsed = current_time - last_log_time
                        if elapsed > 0:
                            recent_chunks = chunk_count
                            chunks_per_sec = recent_chunks / elapsed
                            print(f"Audio generation rate: {chunks_per_sec:.2f} chunks/second")
                        last_log_time = current_time
                        # Reset chunk counter for next interval
                        chunk_count = 0
        except Exception as e:
            print(f"Error in token processing: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            # Always signal completion, even if there was an error
            print("Producer completed - setting done event")
            producer_done_event.set()
            # Add sentinel to queue to signal end of stream
            audio_queue.put(None)

    def run_async():
        """Run the async producer in its own thread"""
        asyncio.run(async_producer())

    # Use a separate thread with higher priority for producer
    thread = threading.Thread(target=run_async, name="TokenProcessor")
    thread.daemon = True  # Allow thread to be terminated when main thread exits
    thread.start()
    
    # Wait for producer to actually start before proceeding
    # This avoids race conditions where we might try to read from an empty queue
    # before the producer has had a chance to add anything
    producer_started_event.wait(timeout=5.0)
    
    # Optimized I/O approach for all systems
    # This approach is simpler and more reliable than separate code paths
    write_buffer = bytearray()
    buffer_max_size = 1024 * 1024  # 1MB max buffer size (adjustable)
    
    # Keep track of the last time we checked for completion
    last_check_time = time.time()
    check_interval = 1.0  # Check producer status every second
    
    # Process audio chunks until we're done
    while True:
        try:
            # Get the next audio chunk with a short timeout
            # This allows us to periodically check status and handle other events
            audio = audio_queue.get(timeout=0.1)
            
            # None marker indicates end of stream