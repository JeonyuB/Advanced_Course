import React, {useEffect, useState} from 'react';
import InfoTable from "./InfoTable.jsx";

import {useDispatch, useSelector} from "react-redux"; //react-Redux 라이브러리의 훅(Hook) 가져옴
import {getClickName, handleInfo} from "../redux/emp/employeeSlice.js";
import {fetchGetEmployee} from "../redux/emp/employeeApi.js";


const initialState = {
    name:'',
    age:'',
    job: '',
    language: '',
    pay: '',
}

const style = {
    width: "60%",
    margin: "0 auto",
    display: "flex",
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    gap: "20px",
    padding: "20px",
    paddingBottom: "30px",
}

//이름 버튼 목록
const Employees = () => {
    const  {infos, clicked, info} =useSelector(state => state.employees); //redux의 employeeSlice안의 객체변수들 가져옴.
    const dispatch = useDispatch();//액션전달 함수

    //clicked 값이 바뀌면 handleInfo()를 dispatch한다.
    useEffect(() => {//컴포넌트 렌더링 시, 사용될 함수.
        // if (!ctrl) return;
        clicked && dispatch(handleInfo()) // (clicked가 truth 일때 dispatch(handleInfo())실행

    },[dispatch,clicked]);//[] : 의존성- dispatch,clicked가 바뀔 때 실행.

    //info 가 변화할 때,  dispatch(fetchGetEmployee()) 됨.
    useEffect(() => {
        // if (!ctrl) return;
        dispatch(fetchGetEmployee());

    },[dispatch,info]);


    return (
        <>
            <div style={style}>
                {infos?.map((info, idx) => (
                    <button
                        key={idx}
                        onClick={() => {dispatch(getClickName(info.name))}}>
                        {info.name}
                    </button>
                ))}
            </div>
            <InfoTable />
        </>
    );
};

export default Employees;