import numpy as np

from stream_manager import StreamManager


def run_streaming_inference(video_obj, yolo, mlti):
    manager = StreamManager()

    while True:
        ret, frame = video_obj.read()
        if not ret:
            break

        results = yolo.track(
            frame,
            persist=True,
            verbose=False
        )

        if (
            len(results) > 0
            and results[0].boxes is not None
            and results[0].boxes.id is not None
        ):
            result = results[0]

            keypoints = result.keypoints.data.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy()

            show = result.plot()

            ready = manager.update(
                keypoints=keypoints,
                track_ids=track_ids
            )

            if ready:
                person_ids = [person["track_id"] for person in ready]

                ready_keypoints = np.stack(
                    [person["keypoints"] for person in ready],
                    axis=0
                )

                prediction = mlti.predict(
                    ready_keypoints,
                    topk=5,
                    person_ids=person_ids
                )

                print(prediction)

        else:
            manager.update(
                keypoints=np.empty((0, 17, 3), dtype=np.float32),
                track_ids=np.empty((0,), dtype=np.int64)
            )
