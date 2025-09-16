import React, {useState} from 'react';
import ModalWindow from '../modal/ModalWindow';

const Sales = () => {
    const [isModalOpen, setIsModalOpen] = useState(false);

    return (
        <>
            <button onClick={() => setIsModalOpen(true)}>cLICK</button>
            <ModalWindow isOpen={isModalOpen} onCancel={()=>setIsModalOpen(false)}
                style={{width: '500px', height: '500px'}}/>
        </>

    );
};

export default Sales;