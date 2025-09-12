import React, {useEffect, useState} from 'react';

const initialState = {
    name:"",
    age:"",
    job:"",
    language: "",
    pay:"",
}

const formStyle = {
    display: "flex",
    flexDirection: "column",
    width: "300px",
    margin: "20px auto",
    padding: "20px",
    border: "1px solid #ccc",
    borderRadius: "10px",
    backgroundColor: "#f9f9f9",
    boxShadow: "0 4px 8px rgba(0, 0, 0, 0.12)",
}

const labelStyle = {
    marginBottom: "10px",
    display: "flex",
    flexDirection: "column",
    fontWeight: "bold",
    color: "#333",
}

const inputStyle = {
    padding: "10px",
    borderRadius: "5px",
    border: "1px solid #ccc",
    fontSize: "14px",
}

const Register = ({handleRegister}) => {
    const [newEmployee, setNewEmployee] = useState(initialState);
    useEffect(() => {
        setNewEmployee(initialState);
    }, [handleRegister])
    const handleChange = e => {
        console.log(e.target);
        const { name, value } = e.target;
        setNewEmployee(prev=>({...prev, [name]: value}));
    }
    const handleSubmit = e => {
        e.preventDefault();
        handleRegister(newEmployee);
    }
    return (
        <>
            <form style={formStyle} onSubmit={handleSubmit}>
                <label style={labelStyle}>
                    이름:
                    <input style={inputStyle} type="text" name="name" value={newEmployee.name} onChange={handleChange} required />
                </label>
                <label style={labelStyle}>
                    나이
                    <input style={inputStyle} type="text" name="age" value={newEmployee.age} onChange={handleChange} required />
                </label>
                <label style={labelStyle}>
                    직업
                    <input style={inputStyle} type="text" name="job" value={newEmployee.job} onChange={handleChange} required />
                </label>
                <label style={labelStyle}>
                    언어
                    <input style={inputStyle} type="text" name="language"  value={newEmployee.language} onChange={handleChange} required />
                </label>
                <label style={labelStyle}>
                    급여
                    <input style={inputStyle} type="text" name="pay" value={newEmployee.pay} onChange={handleChange} required />
                </label>
                <button>제출</button>
            </form>
        </>
    );
};

export default Register;