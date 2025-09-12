package com.example.emp.repository;
import com.example.emp.entity.Employees;
import org.springframework.data.domain.Sort;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.Optional;
import java.util.List;

public interface EmployeesRepository extends JpaRepository<Employees, Long> {
    Optional<Employees> findByName(String name);
    default List<Employees> findAllOrderbyIdDesc(){
        return findAll(Sort.by(Sort.Direction.DESC, "id"));
    }
}
