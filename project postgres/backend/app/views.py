from django.http import HttpRequest, HttpResponse
# from django.shortcuts import render, get_object_or_404
from rest_framework.generics import get_object_or_404
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response

from .models import Employee
from .serializers import EmployeeSerializer


# objects.all()
# objects.get(name=name)
# Employee.save()#값 저장
# delete()#값 삭제


@api_view(['GET', 'POST'])
@permission_classes([AllowAny])
def employees(request):
    #Get방식
    if request.method == 'GET':
        infos =  Employee.objects.all()
        return Response(EmployeeSerializer(infos, many=True).data)

    #Post방식
    resp = request.data #Post방식일 경우 response에 json형식으로 받는다
    print(resp)
    serializer = EmployeeSerializer(data=resp)

    if serializer.is_valid(): #형식이 유효할 경우
        serializer.save() #세이브
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['DELETE'])
@permission_classes([AllowAny])
def employees_delete(request, name):
    emp = get_object_or_404(Employee, name=name)
    print(emp)
    emp.delete()
    return Response(status=status.HTTP_204_NO_CONTENT)

@api_view(['PUT'])
@permission_classes([AllowAny])
def employees_update(request, name):
    emp = get_object_or_404(Employee, name=name)#Employee 모델에서 name이 일치하는 객체를 가져옴(아닐시 404)
    print(emp)
    resp = request.data
    serializer = EmployeeSerializer(emp, data=resp)
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data, status=status.HTTP_200_OK)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)



