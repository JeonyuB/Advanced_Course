import React, {useState, useEffect} from 'react';
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
const initialState = {
    name:"",
    age:"",
    job:"",
    language: "",
    pay:"",
}
const Update = ({clicked, infos,  handleUpdate}) => {
    const [employee, setEmployee] = useState(initialState);

    useEffect(()=>{
        clicked && infos && setEmployee(infos.find(infos => infos.name === clicked))
    }, [infos, clicked])

    const handleChange = e => {
        // console.log(e.target);
        const { name, value } = e.target;
        setEmployee(prev=>({...prev, [name]: value}));
    }
    const handleSubmit = e => {
        e.preventDefault();
        handleUpdate(employee);
    }
    return (
        <>
            <form style={formStyle} onSubmit={handleSubmit}>
                <label style={labelStyle}>
                    이름:
                    <input style={inputStyle} type="text" name="name" value={employee.name} disabled />
                </label>
                <label style={labelStyle}>
                    나이
                    <input style={inputStyle} type="text" name="age" value={employee.age} onChange={handleChange} required />
                </label>
                <label style={labelStyle}>
                    직업
                    <input style={inputStyle} type="text" name="job" value={employee.job} onChange={handleChange} required />
                </label>
                <label style={labelStyle}>
                    언어
                    <input style={inputStyle} type="text" name="language"  value={employee.language} onChange={handleChange} required />
                </label>
                <label style={labelStyle}>
                    급여
                    <input style={inputStyle} type="text" name="pay"  value={employee.pay} onChange={handleChange} required />
                </label>
                <button>제출</button>
            </form>
            
        </>
    );
};

export default Update;