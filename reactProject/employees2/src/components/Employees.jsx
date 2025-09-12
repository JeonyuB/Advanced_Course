import React, {useEffect, useState} from 'react';
import InfoTable from "./InfoTable.jsx";

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

const Employees = ({infos, clicked,  getClickName}) => {
    const [info, setInfo] = useState(initialState);
    const handleClick = (n) => {
        getClickName(n);
        setInfo(infos.find(info=>info.name === n))
    }
    useEffect(() => {
        if(!clicked){
            setInfo(initialState);
            return;
        }
        clicked && infos && setInfo(infos.find(info=>info.name === clicked))
    }, [clicked, infos])

    return (
        <>
            <div style={style}>
                {infos.map((info, idx) => (
                    <button
                        key={idx}
                        onClick={() => {handleClick(info.name)}}>
                        {info.name}
                    </button>
                ))}
            </div>
            <InfoTable info={info}/>
        </>
    );
};

export default Employees;