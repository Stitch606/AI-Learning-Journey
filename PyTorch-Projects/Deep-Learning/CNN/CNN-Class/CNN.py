import torch


class CNN(torch.nn.Module):
    def __init__(self, in_channel, out_channel, karnal_size, num_layers):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.karnal_size = karnal_size
        self.num_layers = num_layers
        self.conv_layers = torch.nn.ModuleList()

        in_channel = self.in_channel

        for i in range(self.num_layers):
            cov = torch.nn.Conv2d(
                in_channel,
                self.out_channel,
                self.karnal_size,
                padding=1,
            )
            self.conv_layers.append(cov)
            in_channel = self.out_channel
            self.in_channel = self.out_channel

            if (i + 1) % 3 == 0:
                self.out_channel = self.out_channel + self.out_channel

        self.dense = torch.nn.Linear(in_channel, 4)

    def forward(self, x):
        pool = torch.nn.MaxPool2d(2)

        for index, layer in enumerate(self.conv_layers):
            x = layer(x)
            x = torch.relu(x)

            if (index + 1) % 5 == 0:
                x = pool(x)

        g_pool = torch.nn.AdaptiveAvgPool2d(1)
        x = torch.flatten(g_pool(x), start_dim=1)
        z = self.dense(x)
        return z


def training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(3, 3, 3, 20).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()

    for i in range(1):
        for img, label in train_data:
            x, y = (img.to(device), label.to(device))
            x = model(x)
            loss = criterion(x, y)
            optim.zero_grad()
            loss.backward()
            optim.step()

            with torch.no_grad():
                for val, lbl in val_data:
                    val, lbl = (val.to(device), lbl.to(device))
                    val = model(val)
                    loss_val = criterion(val, lbl)
                    print("Training :", loss.item())
                    print("Evaluation :", loss_val.item())

    return model
