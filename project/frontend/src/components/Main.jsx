import Employees from "./Employees.jsx";
import Register from "./Register.jsx";
import Update from "./Update.jsx";
import {useDispatch, useSelector} from "react-redux";
import {useEffect} from "react";
import {fetchGetEmployee,fetchDeleteEmployee} from "../redux/emp/employeeAPI.js";
import {handleClick} from "../redux/emp/employeeSlice.js";

const style={
    display:"flex",
    flexDirection:"row",
    justifyContent:"space-between",
    alignItems:"center",
    padding:"20px",
    gap:"10px",
}

const controls=["register","update","delete","reset"]

// state.employees = {
//     infos:[
//         {name:"John",age:34,job:"FrontEnd",language:"React",pay:400},
//         {name:"Susan",age:28,job:"BackEnd",language:"Java",pay:490},
//         {name:"Peter",age:39,job:"FrontEnd",language:"JavaScript",pay:300},
//         {name:"Sue",age:30,job:"PM",language:"Java",pay:500},
//         {name:"Jack",age:31,job:"Publisher",language:"Python",pay:600},
//     ],
//         clicked:"",
//     ctrl:"",
// }

const Main = () => {
    const {ctrl, info, clicked}=useSelector((state) => state.employees);
    const dispatch = useDispatch();
    useEffect(()=>{
        dispatch(fetchGetEmployee());
    },[dispatch, info]);
    const handleControl=(c)=>{
        dispatch(handleClick(c))
        if(c==="delete"){
            dispatch(fetchDeleteEmployee(clicked))
        }
    }

    return (
        <>
            <div>
                <Employees/>
            </div>
            <div style={style}>
                {controls.map((control, index) => (
                    <button key={index} onClick={()=> {
                        handleControl(control)
                    }}>{control}</button>
                ))}
            </div>
            <div>
                {ctrl==="register" && <Register/>}
                {ctrl==="update" && <Update/>}
            </div>
        </>
    );
};

export default Main;