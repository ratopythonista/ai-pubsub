from jai_pubsub.modules.sum import MultiProcessSum
from jai_pubsub.app import foo_task

sum_obj = MultiProcessSum([1, 2, 3, 4, 5, 6])

foo_task.apply_async((sum_obj,), serializer="pickle")
