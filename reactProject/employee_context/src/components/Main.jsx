import React, {useContext, useState} from 'react';
import Employees from "./Employees.jsx";
import Register from "./Register.jsx";
import Update from "./Update.jsx";
import EmployeeContext from "../context/EmployeeContext.jsx";
import useEmployeeContext from "../context/EmployeeContext.jsx";



const style = {
    width: "60%",
    margin: "0 auto",
    display: "flex",
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    gap:"10px",
    padding:"20px",
}
const Main = () => {
   const {controls, ctrl, handleClick} = useEmployeeContext();
    return (
        <>
            <div>
                <Employees/>
            </div>
            <div style={style}>
                {controls.map((control, index) => (
                    <button key={index} onClick={()=>handleClick(control)}>{control}</button>
                ))}
            </div>
            <div>
                {ctrl==="register" && (<Register />)}
                {ctrl==="update" && (
                    <Update />)
                }
            </div>
        </>
    );
};
export default Main;