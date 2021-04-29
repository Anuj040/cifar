"""Module implementing custom model.fit for combined model training"""

import tensorflow.keras.models as KM


class Trainer(KM.Model):
    def __init__(self, combined: KM.Model):
        super(Trainer, self).__init__()
        self.combined = combined

    def compile(self):
        super(Trainer, self).compile()

    def call(self, *args, **kwargs):
        super(Trainer, self).call(*args, **kwargs)

    def train_step(self, *args, **kwargs):
        super(Trainer, self).train_step(*args, **kwargs)


if __name__ == "__main__":
    trainer = Trainer()
