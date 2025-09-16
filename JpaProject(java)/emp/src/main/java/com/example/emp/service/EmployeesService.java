package com.example.emp.service;


import com.example.emp.entity.Employees;
import com.example.emp.repository.EmployeesRepository;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.server.ResponseStatusException;

import java.util.List;


@Service
@Transactional(readOnly = true)
public class EmployeesService {
    private final EmployeesRepository repository;
    public EmployeesService(EmployeesRepository repository) {
        this.repository = repository;
    }

    public List<Employees> findAll() {
        return repository.findAllOrderByIdDesc();
    }

    @Transactional
    public Employees register(Employees employees) {
        Employees emp = new Employees();
        emp.setName(employees.getName());
        emp.setAge(employees.getAge());
        emp.setJob(employees.getJob());
        emp.setLanguage(employees.getLanguage());
        emp.setPay(employees.getPay());
        return repository.save(emp);
    }

    @Transactional
    public Employees update(String name, Employees employees) {
        Employees emp = repository.findByName(name)
                .orElseThrow(()->new ResponseStatusException(HttpStatus.NOT_FOUND, "Employee not found"));
        emp.setName(employees.getName());
        emp.setAge(employees.getAge());
        emp.setJob(employees.getJob());
        emp.setLanguage(employees.getLanguage());
        emp.setPay(employees.getPay());
        return repository.save(emp);
    }

    @Transactional
    public void delete(String name) {
        Employees emp = repository.findByName(name)
                .orElseThrow(()->new ResponseStatusException(HttpStatus.NOT_FOUND, "Employee not found"));
        repository.delete(emp);
    }
}
