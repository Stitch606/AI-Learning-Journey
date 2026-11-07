model = STGCNPP("stgcnpp_jm_ntu60_xsub_hrnet.pth")
mlti = st("stgcnpp_jm_ntu60_xsub_hrnet.pth")
flow = deque(maxlen=60)

i = 0
frames_batch = []

while True:
    i += 1

    arr = np.random.randn(10, 17, 3)
    flow.append(arr)

    if i == 60:
        i = 0

        frames_batch.append(flow.copy())

        if len(frames_batch) == 2:

            batch = np.stack(frames_batch)

            batch = torch.from_numpy(batch).float()
            
            batch = batch.unsqueeze(0)

            result = mlti.predict(batch)

            frames_batch.clear()
            print(result)
