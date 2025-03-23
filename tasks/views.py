from django.shortcuts import render

# Create your views here.
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .models import Task


@csrf_exempt
def add_task(request):
    if request.method == "POST":
        data = json.loads(request.body)
        task = Task.objects.create(
            title=data["title"], description=data.get("description", "")
        )
        return JsonResponse({"message": "Task added", "task_id": task.id})


def list_tasks(request):
    tasks = Task.objects.all().values()
    return JsonResponse(list(tasks), safe=False)
