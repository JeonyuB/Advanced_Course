import {createSlice, createAsyncThunk} from "@reduxjs/toolkit";
import axios from "axios";

export const fetchSaleApi = createAsyncThunk(//무명함수를 받는 콜백함수
    "fetchSaleApi",
    async (_, thunkAPI) => {
        try {
            const response = await axios.get("http://localhost:3001/판매");
            console.log(response.data);
            return response.data;
        }catch (error) {
            return thunkAPI.rejectWithValue(error);
        }

    }
)

const initialState = {
    data:[],
    loading: false,
    error: null,
}

const salesSlice = createSlice({
    name: "sales",
    initialState,
    reducers: {},
    extraReducers:( builder) => {
        builder
            .addCase(fetchSaleApi.pending, (state) => {
                state.loading = true;
            })
            .addCase(fetchSaleApi.fulfilled, (state, action) => {
                state.loading = false
                //console.log(action.payload)
                state.data = action.payload; //action.payload : fetchSaleApi(함수)가 리턴하는 값. +  state.data의 data: initialState의 data: [] 인  data.
            })
            .addCase(fetchSaleApi.rejected, (state, action) => {
                state.loading = false;
                state.error = action.payload;
            });
    },
}
)

export default salesSlice.reducer;