import {createSlice} from "@reduxjs/toolkit";
import {
    fetchPostEmployee,
    fetchGetEmployee,
    fetchDeleteEmployee,
    fetchUpdateEmployee

} from "./employeeAPI.js";

const initialInfo={
    name:"",
    age:"",
    job:"",
    language:"",
    pay:"",
}
const initialState = {
    info:initialInfo,
    clicked:"",
    ctrl:"",
    infos:[],
    loading:false,
    error:false,
}
// Reducer == 함수
// extraReducer : API를 받는 곳
const employeeSlice=createSlice({
    name: "employee",
    initialState,
    reducers: {
        getClickName : (state,action) => {
            // console.log(action)
            state.clicked = action.payload;
        },
        handleClick : (state, action) => {
            // console.log("clicked",action.payload);
            if(action.payload==="delete"){
                state.clicked = "";
                state.info=initialInfo;
                state.ctrl="delete";
                return ;
            }
            if(action.payload==="reset"){
                return state;
            }
            state.ctrl = action.payload;
        },
        // handleRegister:(state,action) => {
        //     if(state.infos.find(info=>info.name===action.payload.name)){
        //         return alert("이미 존재하는 이름입니다! 다른 이름을 사용하세요!")
        //     }
        //     state.infos=[...state.infos,action.payload];
        //     state.clicked = action.payload;
        // },
        // handleUpdate:(state,action) => {
        //     state.infos=state.infos.map((info)=>info.name===action.payload.name
        //     ? action.payload : info)
        // },
        handleInfo:(state) => {
            state.info=state.infos.find(info=>info.name===state.clicked);
        },
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
            .addCase(fetchPostEmployee.fulfilled, (state,action) => {
                state.loading = false;
                const {payload}=action;
                delete payload.id
                state.info = action.payload;
            })
            .addCase(fetchPostEmployee.rejected, (state, action) => {
                state.loading = false;
                state.error = action.payload;
            })
            // PUT (수정)
            .addCase(fetchUpdateEmployee.pending, (state) => {
                state.loading = true;
                state.error = null;
            })
            .addCase(fetchUpdateEmployee.fulfilled, (state, action) => {
                state.loading = false;
                const {payload}=action;
                delete payload.id
                state.info = action.payload;
            })
            .addCase(fetchUpdateEmployee.rejected, (state, action) => {
                state.loading = false;
                state.error = action.payload;
            });
    },
})

export const {getClickName, handleClick, handleInfo} = employeeSlice.actions;
export default employeeSlice.reducer;