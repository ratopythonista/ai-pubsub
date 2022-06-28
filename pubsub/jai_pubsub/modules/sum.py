from time import sleep
import billiard as multiprocessing


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


class MultiProcessSum(JaiTask):
    def __init__(self, number_list: list) -> None:
        self.number_list = number_list

    def compile(self):
        return self

    def run(self):
        with multiprocessing.Pool(5) as p:
            return p.map(self.pow, self.number_list)

    def pow(self, number: int):
        return number**2
        