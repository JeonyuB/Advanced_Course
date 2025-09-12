from django.db import models

# Create your models here.



# Create your views here.

class Employee(models.Model):
    name = models.CharField("이름",max_length=100)
    age = models.IntegerField("나이")
    job = models.CharField("직업",max_length=100)
    language = models.CharField("언어",max_length=100)
    pay = models.IntegerField("급여")
    created_at = models.DateTimeField("등록일", auto_now_add=True)
    updated_at = models.DateTimeField("수정일", auto_now=True)

    class Meta:
        db_table = 'employee'
        ordering = ["-created_at"]
        verbose_name = "직원"
        verbose_name_plural = "직원 목록"
    def __str__(self):
        return f"{self.name} ({self.job}, {self.language})"
        # return self.name