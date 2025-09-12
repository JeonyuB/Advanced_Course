import React, {useState} from 'react';
import Employees from "./Employees.jsx";
import Update from "./Update.jsx";
import Register from "./Register.jsx";

const initialState = [
    {name: "John", age: 35, job: "frontend", language: "React", pay: 400},
    {name: "Peter", age: 28, job: "backend", language: "Java", pay: 500},
    {name: "Sue", age: 38, job: "publisher", language: "javascript", pay: 600},
    {name: "Susan", age: 40, job: "pm", language: "python", pay: 700},
]

const controllers = ["register", "update", "delete", "reset"];

const styles = {
    display: "flex",
    flexDirection: "column",
    justifyContent: "space-between",
}

const Main = () => {
    const [infos, setInfos] = useState(initialState);
    const [clicked, setClicked] = useState("");
    const [control, setControl] = useState("");
    const [ctrl, setCtrl] = useState("");
    const getClickName = (n) => {
        setClicked(n);
    }
    const handleClick= (c) =>{
        if(c==="delete"){
            setInfos(prev=>[...prev.filter]);
            setClicked("");
            setCtrl('');
            return;
        }
        setCtrl(c);
    }
    const handleRegister = (emp)=>{
        if(infos.some(info => info.name === emp.name)){
            return alert("이미 존재하는 이름입니다. 다른 이름을 사용해주세요");
        }
        setInfos(prev=>[...prev,emp]);
        setClicked('');
    }
    const handleUpdate = (emp) =>{
        emp&&setInfos(infos.map(info=>(info.name===clicked?emp:info)))
    }

    const handleClick = (c) => {
        setCtrl(c);
    }
    return (
        <>
            <div>
                <Employees infos={infos} getClickName={getClickName}/>
            </div>
            <div style={styles}>
                {control.map((controllers, index) => (
                    <button key={index}>{control}</button>
                ))}
            </div>
            <div>
                {ctrl==="register" && (<Register />)}
                {ctrl==="update" && (<Update clicked={clicked}/>)}
                <Update clicked={clicked}/>

            </div>

        </>
    );
};

export default Main;