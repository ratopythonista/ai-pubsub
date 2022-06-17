from jai_pubsub.modules.sum import Sum
from jai_pubsub.app import foo_task

sum_obj = Sum(2, 42)

foo_task.apply_async((sum_obj,), serializer="pickle")
