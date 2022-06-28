from jai_pubsub.modules.tensor import TensonMultiprocess
from jai_pubsub.app import foo_task

tm_obj = TensonMultiprocess(batch_size=4)

foo_task.apply_async((tm_obj,), serializer="pickle")
