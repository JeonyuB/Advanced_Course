import React, {useState} from 'react';
import {Button, Modal} from "antd";

const ModalWindow = ({isOpen, onCancel}) => {

    return (
        <>
            <Modal title={null} open={isOpen} onCancel={onCancel} footer={null} centered closeIcon={false}>
                <div style={{textAlign: 'right', marginTop: '10px'}}>
                    <Button onClick={onCancel}>Cancel</Button>
                </div>
            </Modal>
        </>

    );
};

export default ModalWindow;