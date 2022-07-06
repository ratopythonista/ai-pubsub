from celery import Celery

from jai_pubsub.modules import JaiTask

jai = Celery("jai", broker="redis://redis:6379/0", accept_content=["pickle"])


@jai.task
def foo_task(obj: JaiTask):
    return obj.compile().run()

