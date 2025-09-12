import React, {useContext, useEffect, useState} from 'react';
import Employees from "./Employees.jsx";
import Register from "./Register.jsx";
import Update from "./Update.jsx";

import {useDispatch, useSelector} from "react-redux";
import { handleClick} from "../redux/emp/employeeSlice.js";
import {fetchDeleteEmployee, fetchGetEmployee} from "../redux/emp/employeeApi.js";

const controls=["register", "update", "delete", "reset"]

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

// 메인화면 (컴포넌트 모음집)
const Main = () => {
    const {ctrl,clicked,info} = useSelector((state) => state.employees); // ctrl,clicked,info 값을 읽어옴.
    const dispatch = useDispatch();//useDispatch() 함수 가져옴

    // dispatch, info 변경 시 dispatch(fetchGetEmployee()) 실행.
    useEffect(() => {
       dispatch(fetchGetEmployee());
    },[dispatch, info])

    const handleControl=(c)=>{
       console.log(c)
        dispatch(handleClick(c))
        if(c==="delete") {
            dispatch(fetchDeleteEmployee(clicked));
        }
    }

    return (
        <>
            <div>
                <Employees/>
            </div>
            <div style={style}>
                {controls?.map((control, index) => (
                    //action.payload = control;
                    <button key={index} onClick={()=>{(handleControl(control))}}>{control}</button>
                ))}
            </div>
            <div>
                {ctrl==="register" && (<Register />)}
                {ctrl==="update" && ( <Update />)
                }
            </div>
        </>
    );
};
export default Main;