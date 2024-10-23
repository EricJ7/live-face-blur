from video import Video

if __name__=="__main__":
    try:
        stream=Video()
        stream.startStream(g=False)
    except ValueError as e:
        print(e)
    except KeyboardInterrupt:
        print("Stream cancelled")
    finally:
        # ensure after stream that the video is released
        if 'stream' in locals():
           time=stream.releaseVideo()
    
    print(f"elapsed_time= {time:.2f}s")





