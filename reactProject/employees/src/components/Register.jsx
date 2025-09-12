import React from 'react';

const newEmployee = {
    name: "John",
    age: 25,
    job: "",
    language: "",
    pay: "",
}
const formStyles = {

}
const labelStyle={

}

const inputStyle={

}

const Register = ({handleRegister}) => {
    const [newemployee, setNewEmployee] = useState(initialState);
    const handleChange = e => {
        const {name, value} = e.target;
        setNewEmployee(prey=>({...prey,[name]: e.target.value}));
    }
    const handleSubmit = (e)=> {
        e.preventDefault();
        handleRegister(newEmployee);
    }
    return (
        <>
            <form onSubmit={handleSubmit}>
                <label style={labelStyle}>
                    이름
                    <input style={inputStyle} type="text" name="name" onChange={handleChange} required />
                </label>
                <label style={labelStyle}>
                    나이
                    <input style={inputStyle} type="text" name="age" required />
                </label>
                <label style={labelStyle}>
                    직업
                    <input style={inputStyle} type="text" name="job" required />
                </label>
                <label style={labelStyle}>
                    언어
                    <input style={inputStyle} type="text" name="language" required />
                </label>
                <label style={labelStyle}>
                    급여
                    <input style={inputStyle} type="text" name="pay" required />
                </label>
                <button>제출</button>
            </form>
        </>
    );
};

export default Register;