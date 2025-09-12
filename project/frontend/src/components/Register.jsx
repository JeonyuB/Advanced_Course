import {useState} from 'react';
import {useDispatch, useSelector} from "react-redux";
import {fetchPostEmployee} from "../redux/emp/employeeAPI.js";


const initialState = {
    name:"",
    age:"",
    job:"",
    language:"",
    pay:"",
}

const formStyle={
    display:"flex",
    flexDirection:"column",
    width:"400px",
    margin:"20px auto",
    alignItems:'center',
    padding:"20px",
    borderRadius:"5px",
    border:"1px solid #ccc",
    backgroundColor:"#f9f9f9",
    boxShadow:"0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19)",
}

const labelStyle={
    marginBottom:"10px",
    display:"flex",
    flexDirection:"column",
    fontWeight:"bold",
    color:"#333",
}

const inputStyle={
    marginBottom:"10px",
    display:"flex",
    flexDirection:"column",
    width:"300px",
    borderRadius:"5px",
}
const btnStyle={
    marginBottom:"10px",
    display:"flex",
    flexDirection:"column",
    backgroundColor:"yellow",
}

const Register = () => {
    const [newEmployee, setNewEmployee] = useState(initialState);
    const dispatch = useDispatch();
    const {infos} = useSelector((state) => state.employees);
    // useEffect(() => {
    //     setNewEmployee(initialState)
    // },[])
    const handleChange=(e)=>{
        const {name,value}=e.target;
        setNewEmployee(prev=>({...prev,[name]:value}))
    }
    const handleSubmit = e =>{
        e.preventDefault(); // 페이지 이동을 막는것
        if(infos.find(info=>info.name === newEmployee.name)){
            alert("이름이 중복됩니다. 다시 이름 입력하세요! ")
            return;
        }
        dispatch(fetchPostEmployee(newEmployee));
        setNewEmployee(initialState);
    }

    return (
        <>
            <h2>등록절차</h2>
            <form style={formStyle} onSubmit={handleSubmit}>
                <label style={labelStyle}>
                     이름 :&nbsp;
                    <input type="text" name="name" onChange={handleChange} value={newEmployee.name} required style={inputStyle}/>
                </label>
                <label style={labelStyle}>
                     나이 :&nbsp;
                    <input type="text" name="age" onChange={handleChange} value={newEmployee.age} required style={inputStyle} />
                </label>
                <label style={labelStyle}>
                     직업 :&nbsp;
                     <input type="text" name="job" onChange={handleChange} value={newEmployee.job} required style={inputStyle} />
                </label>
                <label style={labelStyle}>
                     언어 :&nbsp;
                     <input type="text" name="language" onChange={handleChange} value={newEmployee.language} required style={inputStyle} />
                </label>
                <label style={labelStyle}>
                     급여 :&nbsp;
                     <input type="text" name="pay" onChange={handleChange} value={newEmployee.pay} required style={inputStyle} />
                </label>
                <button style={btnStyle}>제출</button>
            </form>
        </>
    );
};

export default Register;