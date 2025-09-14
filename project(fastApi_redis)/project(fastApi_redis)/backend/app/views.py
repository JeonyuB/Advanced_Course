from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.generics import get_object_or_404
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from .models import Employee
from .serializers import EmployeeSerializer
#
# objects.all()
# Employee.objects.get(name=name)
# Employee.save()
# delete




# Create your views here.
@api_view(['GET', 'POST'])
@permission_classes([AllowAny])
def employees(request):
    if request.method == 'GET':
        infos = Employee.objects.all()
        return Response(EmployeeSerializer(infos, many=True).data)

    response = request.data
    serializer = EmployeeSerializer(data=response)
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# 단일 객체 조회, 수정, 삭제
@api_view(["GET", "PUT", "PATCH", "DELETE"])
def employee_detail(request, name):
    try:
        emp = Employee.objects.get(name=name)
    except Employee.DoesNotExist:
        return Response({"error": "Employee not found"}, status=status.HTTP_404_NOT_FOUND)

    if request.method == "GET":
        serializer = EmployeeSerializer(emp)
        return Response(serializer.data)

    if request.method in ["PUT", "PATCH"]:
        # PUT은 전체 업데이트, PATCH는 부분 업데이트
        serializer = EmployeeSerializer(emp, data=request.data, partial=(request.method == "PATCH"))
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    if request.method == "DELETE":
        emp.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
