from django.db import models


class KerasModel(models.Model):
    name = models.CharField(max_length=100)
    # Use BinaryField if storing as binary data
    model_file = models.FileField(upload_to='models/')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name
