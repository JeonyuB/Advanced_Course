import React, {useState} from 'react';
import Employees from "./Employees.jsx";
import Register from "./Register.jsx";
import Update from "./Update.jsx";
const initialState = [
    {name: "John", age: 35, job: "frontend", language: "React", pay: 400},
    {name: "Peter", age: 28, job: "backend", language: "Java", pay: 500},
    {name: "Sue", age: 38, job: "publisher", language: "JavaScript", pay: 400},
    {name: "Susan", age: 40, job: "pm", language: "python", pay: 600},
]
const controls = ["register", "update", "delete", "reset"]
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
    const [infos, setInfos] = useState(initialState);
    const [clicked, setClicked] = useState('');
    const [ctrl, setCtrl] = useState("");
    const getClickName = (n) => {
        setClicked(n);
    }
    const handleClick = (c) => {
        if(c==="delete"){
            setInfos(prev=>prev.filter(info=>info.name !== clicked));
            setClicked('');
            setCtrl('')
            return;
        }
        if(c==="reset"){
            setInfos(initialState);
            setClicked('');
            setCtrl('');
            return;
        }
        setCtrl(c);
    }
    const handleRegister = (emp) => {
        if(infos.some(info => info.name === emp.name)){
            return alert("이미 존재하는 이름입니다. 다른 이름을 사용하세요!!!")
        }
        setInfos(prev => [...prev, emp]);
        setClicked(emp.name);
    }
    const handleUpdate = (emp) => {
        console.log("update", emp);
        emp && setInfos(prev => (prev.map(info=>(
            info.name === clicked ? emp : info
        ))))
    }
    return (
        <>
            <div>
                <Employees infos={infos} clicked={clicked} getClickName={getClickName} />
            </div>
            <div style={style}>
                {controls.map((control, index) => (
                    <button key={index} onClick={()=>handleClick(control)}>{control}</button>
                ))}
            </div>
            <div>
                {ctrl==="register" && (<Register handleRegister={handleRegister} />)}
                {ctrl==="update" && (
                    <Update
                        clicked={clicked}
                        infos={infos}
                        handleUpdate={handleUpdate}
                    />)
                }
            </div>
        </>
    );
};
export default Main;