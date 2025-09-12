from django.contrib import admin

from app.models import Employee


# Register your models here.
@admin.register(Employee)
class EmployeeAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'age', 'job', 'language', 'pay')
    ordering = ("-id",)