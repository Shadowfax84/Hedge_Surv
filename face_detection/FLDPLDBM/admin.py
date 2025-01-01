from django.contrib import admin
from .models import KerasModel


@admin.register(KerasModel)
class KerasModelAdmin(admin.ModelAdmin):
    # Display these fields in the admin list view
    list_display = ('name', 'created_at')
    search_fields = ('name',)  # Add search functionality by name
