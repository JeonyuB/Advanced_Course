import { createAsyncThunk } from '@reduxjs/toolkit'
import axios from "axios";

// const url = "http://localhost:8080/app/emp" //8080: 스프링부트 백주소
// const url = "http://3.35.37.174:8000/app/emp" // 8000: 파이썬 백주소

const api = axios.create({
    baseURL: "/app/emp/",
    headers: { "Content-Type": "application/json" },
})

//api 함수
export const fetchGetEmployee = createAsyncThunk(
    "fetchEmployees",
    async (_, thunkAPI)=>{
        try{
            const response = await api.get("")
            return response.data;
        }catch(e){
            return thunkAPI.rejectWithValue("데이터로드 실패");
        }

    }
)

export const fetchPostEmployee = createAsyncThunk(
    "fetchPostEmployee",
    async (emp, thunkAPI)=>{
        try{
            const response = await api.post("", emp)
            //console.log(response.data);
            return response.data;
        }catch(e){
            return thunkAPI.rejectWithValue("데이터 전송 실패");
        }

    }
)

export const fetchDeleteEmployee = createAsyncThunk(
    "fetchDeleteEmployee",
    async (name, thunkAPI)=>{
        // console.log("clicked",name);
        try{
            const response = await api.delete(`${name}`)
            //console.log(response.data);
            return response.data;
        }catch(e){
            return thunkAPI.rejectWithValue("데이터 전송 실패");
        }

    }
)

export const fetchUpdateEmployee = createAsyncThunk(
    "fetchUpdateEmployee",
    async (emp, thunkAPI)=>{
        try{
            const response = await api.put(`${emp.name}/`,emp)
            //console.log(response.data);
            return response.data;
        }catch(e){
            return thunkAPI.rejectWithValue("데이터 수정 실패");
        }

    }
)