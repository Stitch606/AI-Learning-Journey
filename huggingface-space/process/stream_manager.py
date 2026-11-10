from collections import deque
from time import monotonic

import numpy as np


class StreamManager:
    def __init__(
        self,
        sequence_length=100,
        predict_stride=25,
        disappear_timeout=2.0
    ):
        self.sequence_length = sequence_length
        self.predict_stride = predict_stride
        self.disappear_timeout = disappear_timeout
        self.tracks = {}

    def update(self, keypoints, track_ids, timestamp=None):
        if timestamp is None:
            timestamp = monotonic()

        ready = []

        expired = [
            track_id
            for track_id, track in self.tracks.items()
            if timestamp - track["last_seen"] > self.disappear_timeout
        ]

        for track_id in expired:
            del self.tracks[track_id]

        for person_keypoints, track_id in zip(keypoints, track_ids):
            if track_id is None:
                continue

            track_id = int(track_id)

            if track_id not in self.tracks:
                self.tracks[track_id] = {
                    "buffer": deque(maxlen=self.sequence_length),
                    "last_seen": timestamp,
                    "frames_since_prediction": 0,
                    "predicted_once": False
                }

            track = self.tracks[track_id]
            track["last_seen"] = timestamp
            track["buffer"].append(
                np.asarray(person_keypoints, dtype=np.float32).copy()
            )
            track["frames_since_prediction"] += 1

            if len(track["buffer"]) < self.sequence_length:
                continue

            if (
                not track["predicted_once"]
                or track["frames_since_prediction"] >= self.predict_stride
            ):
                ready.append({
                    "track_id": track_id,
                    "keypoints": np.stack(track["buffer"], axis=0)
                })

                track["predicted_once"] = True
                track["frames_since_prediction"] = 0

        return ready

    def reset(self):
        self.tracks.clear()

    def get_active_tracks(self):
        return list(self.tracks.keys())
