// src/redux/employeeAPI.js
import { createAsyncThunk } from "@reduxjs/toolkit";
import axios from "axios";

/**
 * 베이스 URL 결정 규칙
 * - 배포(Nginx 프록시): 기본 "/app/emp/"
 * - 로컬 개발 등 환경별로 바꾸고 싶으면 .env(.env.production)에서 VITE_API_BASE 지정
 *   예) VITE_API_BASE=/app/emp/
 */
const BASE = (import.meta?.env?.VITE_API_BASE ?? "/app/emp/").replace(/([^/])$/, "$1/");

/**
 * axios 인스턴스
 * - 타임아웃/JSON 헤더
 * - 응답/에러 인터셉터로 data만 반환, 에러 메시지 통일
 * - (CSRF가 필요하면 withCredentials: true + CSRF 헤더 추가)
 */
const api = axios.create({
  baseURL: BASE,
  timeout: 15000,
  headers: { "Content-Type": "application/json" },
  // withCredentials: true, // CSRF 쿠키를 쓸 때만 활성화
});

api.interceptors.response.use(
  (res) => res.data,
  (err) => {
    const msg =
      err.response?.data?.message ||
      err.response?.data?.detail ||
      err.message ||
      "요청 처리 중 오류가 발생했습니다.";
    return Promise.reject(new Error(msg));
  }
);

/** 유틸: 이름 파라미터(한글, 공백 등) 안전 인코딩 + DRF 트레일링 슬래시 보정 */
const detailPath = (name) => `${encodeURIComponent(name)}/`;

/* =========================
 * Thunks
 * ========================= */

// 전체 조회
export const fetchGetEmployee = createAsyncThunk(
  "employee/fetchGetEmployee",
  async (_, thunkAPI) => {
    try {
      // signal 지원: 컴포넌트 언마운트 시 abort 가능
      const { signal } = thunkAPI;
      return await api.get("", { signal }); // → GET /app/emp/
    } catch (e) {
      return thunkAPI.rejectWithValue(e.message);
    }
  }
);

// 단건 조회 (선택: 필요하면 사용)
export const fetchGetEmployeeByName = createAsyncThunk(
  "employee/fetchGetEmployeeByName",
  async (name, thunkAPI) => {
    try {
      const { signal } = thunkAPI;
      return await api.get(detailPath(name), { signal }); // → GET /app/emp/<name>/
    } catch (e) {
      return thunkAPI.rejectWithValue(e.message);
    }
  }
);

// 등록
export const fetchPostEmployee = createAsyncThunk(
  "employee/fetchPostEmployee",
  async (emp, thunkAPI) => {
    try {
      const { signal } = thunkAPI;
      return await api.post("", emp, { signal }); // → POST /app/emp/
    } catch (e) {
      return thunkAPI.rejectWithValue(e.message);
    }
  }
);

// 수정(전체 교체)
export const fetchUpdateEmployee = createAsyncThunk(
  "employee/fetchUpdateEmployee",
  async (emp, thunkAPI) => {
    try {
      if (!emp?.name) throw new Error("수정하려는 직원의 name이 없습니다.");
      const { signal } = thunkAPI;
      return await api.put(detailPath(emp.name), emp, { signal }); // → PUT /app/emp/<name>/
    } catch (e) {
      return thunkAPI.rejectWithValue(e.message);
    }
  }
);

// 부분 수정(PATCH) — 필요 시 사용
export const fetchPatchEmployee = createAsyncThunk(
  "employee/fetchPatchEmployee",
  async ({ name, patch }, thunkAPI) => {
    try {
      if (!name) throw new Error("수정하려는 직원의 name이 없습니다.");
      const { signal } = thunkAPI;
      return await api.patch(detailPath(name), patch, { signal }); // → PATCH /app/emp/<name>/
    } catch (e) {
      return thunkAPI.rejectWithValue(e.message);
    }
  }
);

// 삭제
export const fetchDeleteEmployee = createAsyncThunk(
  "employee/fetchDeleteEmployee",
  async (name, thunkAPI) => {
    try {
      if (!name) throw new Error("삭제하려는 직원의 name이 없습니다.");
      const { signal } = thunkAPI;
      return await api.delete(detailPath(name), { signal }); // → DELETE /app/emp/<name>/
    } catch (e) {
      return thunkAPI.rejectWithValue(e.message);
    }
  }
);
