import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM


class Cifar:
    def __init__(self) -> None:
        self.model = self.build()

    def encoder(self) -> KM.Model:
        ...

    def decoder(self) -> KM.Model:
        ...

    def build(self) -> KM.Model:
        return None


if __name__ == "__main__":
    model = Cifar
