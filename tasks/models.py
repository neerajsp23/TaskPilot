from django.db import models

# Create your models here.
from django.db import models


class Task(models.Model):
    task = models.CharField(max_length=255)
    priority = models.CharField(blank=True, null=True)
    description = models.TextField(blank=True, null=True)
    scheduled_time = models.DateTimeField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.task
