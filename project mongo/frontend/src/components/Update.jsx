import React, {useState, useEffect} from 'react';
import {useDispatch, useSelector} from "react-redux";
// import {handleUpdate} from "../redux/emp/employeeSlice.js";
import {fetchUpdateEmployee} from "../redux/emp/employeeApi.js";

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
const Update = () => {
    const {clicked, infos} = useSelector(state => state.employees); //기존의 정보를 불러오기 위함
    const dispatch = useDispatch();
    const [modi_employee, setEmployee] = useState(initialState); //수정한 값 저장할 곳

    //이름 클릭 시, 기존의 저장된 정보를 가져오는 역할
    useEffect(()=>{
        // infos?.length : 여기서 infos? 의 ?는 null 또는 undefined이 아닌 경우 메서드가 실행하게끔 하는 것이다.
        if( clicked && infos?.length){
            setEmployee(infos.find(infos => infos.name === clicked));
        }

    }, [infos, clicked])

    const handleChange = e => {
        // console.log(e.target);
        const { name, value } = e.target;
        setEmployee(prev=>({...prev, [name]: value}));
    }
    const handleSubmit = e => {
        e.preventDefault();
        dispatch(fetchUpdateEmployee(modi_employee));
    }
    return (
        <>
            <form style={formStyle} onSubmit={handleSubmit}>
                <label style={labelStyle}>
                    이름:
                    <input style={inputStyle} type="text" name="name" value={modi_employee.name} disabled />
                </label>
                <label style={labelStyle}>
                    나이
                    <input style={inputStyle} type="text" name="age" value={modi_employee.age} onChange={handleChange} required />
                </label>
                <label style={labelStyle}>
                    직업
                    <input style={inputStyle} type="text" name="job" value={modi_employee.job} onChange={handleChange} required />
                </label>
                <label style={labelStyle}>
                    언어
                    <input style={inputStyle} type="text" name="language"  value={modi_employee.language} onChange={handleChange} required />
                </label>
                <label style={labelStyle}>
                    급여
                    <input style={inputStyle} type="text" name="pay"  value={modi_employee.pay} onChange={handleChange} required />
                </label>
                <button>제출</button>
            </form>
            
        </>
    );
};

export default Update;