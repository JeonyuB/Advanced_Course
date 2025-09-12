import { createSlice } from '@reduxjs/toolkit'
import {fetchGetEmployee, fetchPostEmployee, fetchUpdateEmployee} from "./employeeApi.js";


const initialInfo={name: "", age: "", job: "", language: "", pay: ""}

const initialState ={
    infos:[],
    clicked: "",
    ctrl: "",
    info:{name: "", age: "", job: "", language: "", pay: ""},
    loading: false,
    error: false
}

const employeeSlice = createSlice({
    name: "employeeSlice",
    initialState,
    reducers: {
        getClickName:(state, action)=>{
            state.clicked = action.payload;
        },
        handleClick: (state, action) =>{
            if(action.payload === "delete"){
                // state.infos = state.infos.filter(info=>info.name !== state.clicked)
                state.clicked = ""
                state.ctrl= "delete"
                state.info = initialInfo;
                return;
            }
            if(action.payload ==="reset"){
                // return initialState;
                return {...state};
            }
            state.ctrl = action.payload;
        },
        // handleRegister: (state, action)=>{
        //     if(state.infos.some(info => info.name === action.payload.name)){
        //         return alert("이미 존재하는 이름입니다. 다른 이름을 사용하세요!!!")
        //     }
        //     state.infos = [...state.infos, action.payload]
        //     state.clicked = action.payload.name;
        // },
        // handleUpdate:(state, action) => {
        //     state.infos = state.infos.map(info => info.name === action.payload.name ?
        //         action.payload : info // 이름이 일치하는 직원 정보 업데이트, 아닐 시 정보 그대로 유지.
        //     )
        // },
        handleInfo: (state)=>{
            //전체 직원의 목록(state.infos)에서 clicked된 직원의 이름(info.name)과 같은 직원을 찾는다.
            state.info = state.infos.find(info=>info.name === state.clicked)
        }

    },
    extraReducers: (builder) => {
        builder
            // fetchGet 처리
            .addCase(fetchGetEmployee.pending, (state) => {
                state.loading = true;
                state.error = null;
            })
            .addCase(fetchGetEmployee.fulfilled, (state, action) => {
                state.loading = false;
                state.infos = action.payload;
            })
            .addCase(fetchGetEmployee.rejected, (state, action) => {
                state.loading = false;
                state.error = action.payload;
            })

            // fetchPost 처리
            .addCase(fetchPostEmployee.pending, (state) => {
                state.loading = true;
                state.error = null;
            })
            .addCase(fetchPostEmployee.fulfilled, (state, action) => {
                state.loading = false;
                const {payload} = action;
                delete payload.id;
                state.info = payload;
            })
            .addCase(fetchPostEmployee.rejected, (state, action) => {
                state.loading = false;
                state.error = action.payload;
            })

            // fetchUpdate 처리
            .addCase(fetchUpdateEmployee.pending, (state) => {
                state.loading = true;
                state.error = null;
            })
            .addCase(fetchUpdateEmployee.fulfilled, (state, action) => {
                state.loading = false;
                const { payload } = action;
                delete payload.id;
                state.info = payload;
            })
            .addCase(fetchUpdateEmployee.rejected, (state, action) => {
                state.loading = false;
                state.error = action.payload;
            })
    },
})

export const {
    getClickName,
    handleRegister,
    handleClick,
    handleUpdate,
    handleInfo,
} = employeeSlice.actions;
export default employeeSlice.reducer;