import { createAsyncThunk } from "@reduxjs/toolkit";
import axios from "axios";

// axios 인스턴스 생성
const api = axios.create({
    baseURL: "http://52.79.163.49/app/emp",
    headers: {
        "Content-Type": "application/json"
    }
});

// 전체 조회
export const fetchGetEmployee = createAsyncThunk(
    "employee/fetchGetEmployee",
    async (_, thunkAPI) => {
        try {
            const resp = await api.get("");
            return resp.data;
        } catch (e) {
            return thunkAPI.rejectWithValue(e.response?.data?.message || "데이터 로드 실패");
        }
    }
);

// 등록
export const fetchPostEmployee = createAsyncThunk(
    "employee/fetchPostEmployee",
    async (emp, thunkAPI) => {
        try {
            const resp = await api.post("", emp);
            return resp.data;
        } catch (e) {
            return thunkAPI.rejectWithValue(e.response?.data?.message || "데이터 전송 실패");
        }
    }
);

// 삭제
export const fetchDeleteEmployee = createAsyncThunk(
    "employee/fetchDeleteEmployee",
    async (name, thunkAPI) => {
        try {
            const resp = await api.delete(`${name}`);
            return resp.data;
        } catch (e) {
            return thunkAPI.rejectWithValue(e.response?.data?.message || "데이터 삭제 실패");
        }
    }
);

// 수정
export const fetchUpdateEmployee = createAsyncThunk(
    "employee/fetchUpdateEmployee",
    async (emp, thunkAPI) => {
        try {
            const resp = await api.put(`${emp.name}`, emp);
            return resp.data;
        } catch (e) {
            return thunkAPI.rejectWithValue(e.response?.data?.message || "데이터 수정 실패");
        }
    }
);
