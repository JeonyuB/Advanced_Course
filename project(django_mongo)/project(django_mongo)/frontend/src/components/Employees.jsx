import React, {useEffect} from 'react';
import InfoTable from "./InfoTable.jsx";
import {handleInfo,getClickName} from "../redux/emp/employeeSlice.js";
import {useDispatch, useSelector} from "react-redux";
import {fetchGetEmployee} from "../redux/emp/employeeAPI.js";

const styles={
    display: 'flex',
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    gap:"15px",
    padding:"20px",
}

const Employees = () => {
    const {infos,info, clicked}=useSelector((state)=>state.employees);
    const dispatch = useDispatch();
    // const handleClick = (N) => {
    //     dispatch(getClickName(N));
    //     setInfo(infos.find(info => info.name === N))
    // }
    useEffect(() => {
        // dispatch(getClickName());   이 부분이 잘 못된 부분입니다.
       if(clicked){
           dispatch(handleInfo());
       }
    },[clicked, dispatch])

    useEffect(()=>{
        dispatch(fetchGetEmployee());
    },[dispatch, info])
    return (
        <>
            <div style={styles}>
                {infos.map((info,idx) => (
                    <button
                        key={idx}
                        onClick={()=>{dispatch(getClickName(info.name))}}
                    >
                        {info.name}
                    </button>
                ))}
            </div>
            <InfoTable/>
        </>
    );
};

export default Employees;