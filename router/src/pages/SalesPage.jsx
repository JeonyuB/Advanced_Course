import React, {useEffect, useMemo, useState} from 'react';
import {Link} from "react-router-dom";
import {useDispatch, useSelector} from "react-redux";
import 'ag-grid-community/styles/ag-grid.css';
import 'ag-grid-community/styles/ag-theme-alpine.css'
import {AllCommunityModule} from "ag-grid-community";
import {AgGridReact} from "ag-grid-react";
import {ModuleRegistry} from "ag-grid-community";
import {fetchSaleApi} from "../redux/slices/salesSlice.js";
import {Button} from "antd";
import SaleModal from "../modal/SaleModal";
ModuleRegistry.registerModules([AllCommunityModule])


function SalesPage() {
    const [isModalOpen, setIsModalOpen] = useState(false);
    const dispatch = useDispatch();
    const {loading, error, data} = useSelector(state => state.sales);

    useEffect(() => {
        console.log("use effect")
        dispatch(fetchSaleApi());
    }, [dispatch]);

    const rowData = useMemo(()=>data ?? [], [data])
    const columnDefs = useMemo(()=>[
        {headerName: "주문일", field: "날짜", flex:1},
        {headerName: "제품번호", field: "제품코드", flex:1},
        {headerName: "고객번호", field: "고객코드", flex:1},
        {headerName: "프로모션번호", field: "프로모션코드", flex:1},
        {headerName: "채널번호", field: "채널코드", flex:1},
        {headerName: "수량", field: "Quantity", flex:1},
        {headerName: "단가", field: "UnitPrice", flex:1},
        {headerName: "지역", field: "지역", flex:1},
    ],[])
    if (loading) return <div>Loading...</div>;
    if (error) return <div>Error: {error.message}</div>;

    return (
        <>
        <div style={{display:"flex", justifyContent: "flex-end", paddingBottom:"10px"}}>
            <Button type="primary" onClick={() => setIsModalOpen(false)}>판매등록</Button>
        </div>
        <SaleModal open={isModalOpen} onCancel={() => setIsModalOpen(false)}/>
        <div className="ag-theme-alpine" style={{width: '100%', height: '70vh'}}>
            <AgGridReact rowData={rowData} columnDefs={columnDefs}
                         animateRows={true} domLayout="autoHeight"
                         pagination={true} rowSelection="single"
                         enableBrowserTooltips = {true}
            />
        </div>
        </>
    );
}

export default SalesPage;