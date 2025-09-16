package com.example.emp.entity;

import jakarta.persistence.*;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Entity
@Table(name="employees")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
public class Employees {

    @Id
    @GeneratedValue(strategy = GenerationType.AUTO)
    private int id;

    @NotNull
    @NotBlank
    @Column(nullable = false, length = 100)
    private String name;

    @NotNull
    private int age;

    @NotBlank
    @Column(nullable = false, length = 100)
    private String job;

    @NotBlank
    @Column(nullable = false, length = 100)
    private String language;

    @NotNull
    private int pay;

}

//jpa 기본세팅
