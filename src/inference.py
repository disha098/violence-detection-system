import numpy as np
from src.preprocessing import extract_all_frames

# 🔥 Merge continuous segments
def merge_segments(results, threshold=0.5):

    merged = []
    current = None

    for r in results:
        if r["violence_prob"] > threshold:

            if current is None:
                current = r.copy()
            else:
                current["end_frame"] = r["end_frame"]

        else:
            if current is not None:
                merged.append(current)
                current = None

    if current is not None:
        merged.append(current)

    return merged


# 🔥 Main prediction with timestamps
def predict_with_timestamps(model, video_path, sequence_length=16):

    frames = extract_all_frames(video_path)

    if len(frames) == 0:
        return None

    results = []

    for i in range(0, len(frames) - sequence_length, sequence_length):

        chunk = frames[i:i+sequence_length]
        chunk = np.expand_dims(chunk, axis=0)

        prediction = model.predict(chunk)[0]

        results.append({
            "start_frame": i,
            "end_frame": i + sequence_length,
            "violence_prob": float(prediction[1])
        })

    # 🔥 Merge repetitive segments
    merged = merge_segments(results)

    return merged