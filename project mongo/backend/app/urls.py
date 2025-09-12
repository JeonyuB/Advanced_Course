from django.urls import path
from . import views

#url, uri : request하는 장소
urlpatterns = [
    path("emp/", views.employees, name="employees"),
    path("emp/<str:name>", views.employees_delete, name="employees_delete"),
    path("emp/<str:name>/", views.employees_update, name="employees_update"),
    # 앞 url 호출 시 views.py의 employees_update함수 실행. name="employees_update"으로 경로 참조
]