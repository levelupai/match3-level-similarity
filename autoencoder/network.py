from torch import nn


class AutoEncoderNetwork(nn.Module):

    def __init__(self, encoder_needed=True, decoder_needed=True):
        super().__init__()
        self.encoder_needed = encoder_needed
        self.decoder_needed = decoder_needed

        if encoder_needed:
            # Encoder layers
            self.encoder = nn.Sequential(
                nn.Linear(35 * 12 * 12, 2500),
                nn.ReLU(True),
                nn.Linear(2500, 1000),
                nn.ReLU(True),
                nn.Linear(1000, 500),
                nn.ReLU(True),
                nn.Linear(500, 10),
            )

        if decoder_needed:
            # Decoder layers
            self.decoder = nn.Sequential(
                nn.Linear(10, 500),
                nn.ReLU(True),
                nn.Linear(500, 1000),
                nn.ReLU(True),
                nn.Linear(1000, 2500),
                nn.ReLU(True),
                nn.Linear(2500, 35 * 12 * 12),
            )

    def forward(self, x):
        if self.encoder_needed:
            # Encode
            x = self.encoder(x.reshape((x.size(0), -1)))

        if self.decoder_needed:
            # Decode
            x = self.decoder(x)
            x = x.reshape((-1, 35, 12, 12))

        return x

    def set_encoder_needed(self, encoder_needed=True):
        self.encoder_needed = encoder_needed

    def set_decoder_needed(self, decoder_needed=True):
        self.decoder_needed = decoder_needed
