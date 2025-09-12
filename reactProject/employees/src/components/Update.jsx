import React, {useEffect, useEffect} from 'react';
const formStyles = {

}
const labelStyle={

}

const inputStyle={

}

const initialStyles={


}
const Update = ({clicked, handleUpdate}) => {}
    const [employees, setEmployees] = useState([])

const handleRegister = (emo)=>{
    setInfos(prev=>[...pev,emp]);
}
const handleUpdate = (emp) =>{
        setInfos(infos.map(info=>(info.name===clicked?emp:info)))
}
    useEffect(() => {

    })
    const [infos, setInfos] = useState([])
    const [updateInfos, setUpdateInfos] = useState([])


    return (
        <>
            <form>
                <label>
                    이름
                    <input type="text" value={employee.age} onChange={{handle}} name="name" required />
                </label>
                <label>
                    나이
                    <input type="text" name="age" required />
                </label>
                <label>
                    직업
                    <input type="text" name="job" required />
                </label>
                <label>
                    언어
                    <input type="text" name="language" required />
                </label>
                <label>
                    급여
                    <input type="text" name="pay" required />
                </label>
                <button>제출</button>
            </form>
            
        </>
    );
};

export default Update;