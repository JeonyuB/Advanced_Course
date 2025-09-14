import React from "react";
import { useSelector } from "react-redux";

const tableStyle = {
    width: "700px",
    margin: "0 auto",
    borderCollapse: "collapse",
    fontFamily: "'Roboto', 'Helvetica', 'Arial', sans-serif",
    borderRadius: "8px",
    overflow: "hidden",
    tableLayout: "fixed",
    userSelect: "none",
};

const thStyle = {
    backgroundColor: "#f2f2f2",
    color: "#333",
    textAlign: "left",
    padding: "12px 15px",
    borderBottom: "2px solid #ddd",
    fontWeight: "bold",
    textTransform: "uppercase",
    fontSize: "0.875rem",
};

const tdStyle = {
    padding: "12px 15px",
    borderBottom: "1px solid #eee",
    textAlign: "left",
    color: "#555",
};

const InfoTable = () => {
    const { info } = useSelector((state) => state.employees);

    // ✅ info가 없거나 객체가 아닌 경우 방어
    if (!info || typeof info !== "object") {
        return <div style={{ textAlign: "center", marginTop: "20px" }}>데이터가 없습니다.</div>;
    }

    const keys = Object.keys(info);
    const values = Object.values(info);

    return (
        <table style={tableStyle}>
            <thead>
            <tr>
                {keys.map((key, idx) => (
                    <th key={idx} style={thStyle}>
                        {key}
                    </th>
                ))}
            </tr>
            </thead>
            <tbody>
            <tr>
                {values.map((value, idx) => (
                    <td key={idx} style={tdStyle}>
                        {value}
                    </td>
                ))}
            </tr>
            </tbody>
        </table>
    );
};

export default InfoTable;
