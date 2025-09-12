import React, {useMemo, useState, createContext, useContext} from 'react';

const initialState = [
    {name: "John", age: 35, job: "frontend", language: "React", pay: 400},
    {name: "Peter", age: 28, job: "backend", language: "Java", pay: 500},
    {name: "Sue", age: 38, job: "publisher", language: "JavaScript", pay: 400},
    {name: "Susan", age: 40, job: "pm", language: "python", pay: 600},
]

const EmployeeContext = createContext()

export const EmployeeProvider = ({children}) => {
    const [infos, setInfos] = useState(initialState);
    const [clicked, setClicked] = useState('');
    const [ctrl, setCtrl] = useState("");
    //배열, 객체를 선언할 때 useMemo로 감싼다(userMemo hook을 사용)
    const controls = useMemo(()=>(["register", "update", "delete", "reset"]), [] );

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
    
    //object형식이라 중괄호
    //배열, 객체를 선언할 때 useMemo로 감싼다(userMemo hook을 사용)
    const value = useMemo(()=>({infos, clicked, ctrl, controls, getClickName, handleClick, handleRegister, handleUpdate}),[infos, clicked, ctrl, getClickName, handleClick, handleRegister, handleUpdate]);

    //여기서의 {value}는 바로 위의 value
    //{children}에 main 들어감
    return (
        <EmployeeContext.Provider value={value}>
            {children}
        </EmployeeContext.Provider>
    );
};

//계속되는 선언 방지하기 위해 가져오는 함수 생성
const useEmployeeContext = () => {
    //context = {infos, clicked, ctrl, getClickName, handleClick, handleRegister, handleUpdate}
   //위 요소들(context에 들어간거)을 하나씩 잡아옴
    const context = useContext(EmployeeContext);
    return context;
}

export default useEmployeeContext;