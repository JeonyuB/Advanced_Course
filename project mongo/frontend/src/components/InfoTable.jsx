import React from 'react';
import {useSelector} from "react-redux";

const tableStyle = {
    width: "60%",
    margin: "0 auto",
    borderCollapse: "collapse",
    fontFamily: "'Roboto', 'Helvetica', 'Arial', sans-serif",
    borderRadius: "8px",
    overflow: "hidden",
    tableLayout: "fixed",
}

const thStyle = {
    backgroundColor: "#f2f2f2",
    color:"#333",
    padding: "12px 15px",
    textAlign: "left",
    borderBottom: "2px solid #ddd",
    fontWeight: "bold",
    textTransform: "uppercase",
    fontSize: "0.875rem",
}

const tdStyle = {
    padding: "12px 15px",
    borderBottom: "1px solid #eee",
    textAlign: "left",
    color: "#555",
}

// 세부정보 들어가는 테이블
export const InfoTable = () => {
    const {info} = useSelector((state) => state.employees);

    return (
        <>
            <table style={tableStyle} >
                <thead>
                    <tr>
                        {info&&Object.keys(info).map((key, idx)=>(
                            <th style={thStyle} key={idx}>{key}</th>))}
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        {info&&Object.values(info).map((value, idx)=>(
                            <th style={tdStyle} key={idx}>{value}</th>))}
                    </tr>
                </tbody>
            </table>
            
        </>
    );
};

export default InfoTable;