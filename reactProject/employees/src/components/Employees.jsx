import React, {useState} from 'react';
import InfoTable from "./InfoTable.jsx";

const initialState = {
    name:'',
    age:'',
    job:'',
    language:'',
    pay:'',
}

const styles = {
    display: "flex",
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    gap:"10px",
    padding: "20px",
    paddingBottom: "30px",
}

const Employees = ({infos, getClickName}) => {
    const [info, setInfo] = useState(initialState);
    const handleClick = (n) => {
        getClickName(n);
        setInfo(infos.find(info => info.name === n));
    }

    return (
        <>
            <div style={styles}>
                {infos.map((info, idx) => (
                    <button
                        key={idx}
                        onClick={() => {handleClick(info.name) }}>
                        {info.name}
                    </button>
                ))}
            </div>

            <InfoTable info={info} />
        </>
    );
};

export default Employees;