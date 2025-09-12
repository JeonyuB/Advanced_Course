package com.example.emp.entity;
import jakarta.persistence.*;
import jakarta.validation.constraints.NotBlank;
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

    @Id@GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @NotBlank@Column(nullable = false, length = 100)
    private String name;

    @Column(nullable = false)
    private int age;

    @NotBlank@Column(nullable = false, length = 100)
    private String job;

    @NotBlank@Column(nullable = false, length = 100)
    private String language;

    @Column(nullable = false)
    private int pay;
}
