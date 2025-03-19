import lightning as L
import torch
import torch.nn.functional as F

class Autoencoder(L.LightningModule):
    def __init__(
        self,
        n_input_channels=8,
        patch_size=9,
        embedding_size=64, # Change into 64 embedding size
        optimizer_config=None,
    ):
        """
        A convolutional autoencoder that encodes (batch_size, n_input_channels, patch_size, patch_size)
        patches into a latent embedding, then decodes them back to the original shape.

        Args:
            n_input_channels (int): Number of input channels. Default = 8 (from the lab instructions).
            patch_size (int): Width/height of the input patches (e.g., 9).
            embedding_size (int): Dimension of the latent embedding.
            optimizer_config (dict): Hyperparameters for the optimizer (learning rate, etc.).
        """
        super().__init__()
        if optimizer_config is None:
            optimizer_config = {}
        self.optimizer_config = optimizer_config

        # Encoder
        # ------------------------
        # For a 9x9 patch, apply two convolution layers that reduce the spatial dimensions.
        # After these, flatten and do a linear to get to 'embedding_size'.
        self.encoder_cnn = torch.nn.Sequential(
            # in: (batch_size, 8, 9, 9)
            torch.nn.Conv2d(in_channels=n_input_channels, out_channels=16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            # (batch_size, 16, 9, 9)

            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            # (batch_size, 32, 5, 5) because stride=2 roughly halves 9 -> 5
        )
        # Flatten and go to embedding_size
        self.encoder_fc = torch.nn.Sequential(
            torch.nn.Linear(32 * 5 * 5, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, embedding_size),
        )

        # Decoder
        # ------------------------
        # 1) Map from embedding_size back up to (32, 5, 5)
        # 2) Use transposed convolutions to upsample back to (8, 9, 9)
        self.decoder_fc = torch.nn.Sequential(
            torch.nn.Linear(embedding_size, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32 * 5 * 5),
            torch.nn.ReLU()
        )

        self.decoder_cnn = torch.nn.Sequential(
            # un-flatten to (32, 5, 5)
            # (batch_size, 32, 5, 5) -> upsample to (batch_size, 16, 9, 9)
            torch.nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3,
                                     stride=2, padding=1, output_padding=0),
            torch.nn.ReLU(),
            # now (batch_size, 16, 9, 9)

            # final layer: reduce channels back to 8
            torch.nn.ConvTranspose2d(in_channels=16, out_channels=n_input_channels,
                                     kernel_size=3, stride=1, padding=1),
            # could use a sigmoid or tanh if data is normalized; here we leave it as raw output
        )

    def forward(self, x):
        """
        Forward pass: encode, then decode.
        Args:
            x (Tensor): shape (batch_size, n_input_channels, patch_size, patch_size)
        Returns:
            Reconstructed tensor of the same shape as x
        """
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded

    def encode(self, x):
        """
        Pass input through the CNN encoder and FC layers to produce embeddings.
        """
        # CNN part
        z = self.encoder_cnn(x)
        # Flatten
        z = z.view(z.size(0), -1)
        # FC part
        z = self.encoder_fc(z)
        return z

    def decode(self, z):
        """
        Map embedding z back to the original (n_input_channels, patch_size, patch_size) shape.
        """
        # FC part
        z = self.decoder_fc(z)
        # Reshape to (batch_size, 32, 5, 5)
        z = z.view(z.size(0), 32, 5, 5)
        # Transposed CNN part
        out = self.decoder_cnn(z)
        return out

    def training_step(self, batch, batch_idx):
        """
        Training step for the autoencoder. MSE loss between x and reconstructed x.
        """
        x = batch
        x_hat = self.forward(x)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the autoencoder. MSE loss between x and reconstructed x.
        """
        x = batch
        x_hat = self.forward(x)
        loss = F.mse_loss(x_hat, x)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        """
        Set up the optimizer (Adam by default).
        """
        optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_config)
        return optimizer

    def embed(self, x):
        """
        Returns the latent embedding (batch_size, embedding_size).
        Useful for downstream tasks or transfer learning.
        """
        return self.encode(x)
