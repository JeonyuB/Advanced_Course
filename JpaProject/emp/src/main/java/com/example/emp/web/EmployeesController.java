package com.example.emp.web;


import com.example.emp.entity.Employees;
import com.example.emp.service.EmployeesService;
import jakarta.validation.Valid;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/app/emp")
public class EmployeesController {
    private final EmployeesService service;
    public EmployeesController(EmployeesService service){
        this.service = service;
    }

    @GetMapping
    public List<Employees> findAll()
    {
        return service.findAll();
    }

    @PostMapping
    @ResponseStatus(HttpStatus.CREATED)
    public Employees create(@Valid @RequestBody Employees employees){
        Employees emp = service.register(employees);
        return new Employees(
                emp.getId(),
                emp.getName(),
                emp.getAge(),
                emp.getJob(),
                emp.getLanguage(),
                emp.getPay()
        );
    }

    @PutMapping("/{name}")
    public Employees update(@PathVariable String name, @Valid @RequestBody Employees employees){
        Employees emp = service.update(name, employees);
        return new Employees(
                emp.getId(),
                emp.getName(),
                emp.getAge(),
                emp.getJob(),
                emp.getLanguage(),
                emp.getPay()
        );

    }

    @DeleteMapping("/{name}")
    @ResponseStatus(HttpStatus.NO_CONTENT)
    public void delete(@PathVariable String name){
        service.delete(name);
    }

}
