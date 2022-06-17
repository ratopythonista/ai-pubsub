from time import sleep


from jai_pubsub.modules import JaiTask


class Sum(JaiTask):
    def __init__(self, a: int, b: int) -> None:
        self.a = a
        self.b = b

    def compile(self):
        sleep(10)
        return self

    def run(self):
        return self.a + self.b
