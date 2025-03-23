from django.urls import path
from .views import add_task, list_tasks

urlpatterns = [
    path("add/", add_task, name="add_task"),
    path("list/", list_tasks, name="list_tasks"),
]
