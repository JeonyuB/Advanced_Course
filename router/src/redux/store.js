import {configureStore} from "@reduxjs/toolkit";
import sales from "./slices/salesSlice.js";

const store = configureStore({//키값을 같이
    reducer: {
        sales,
    }
})

export default store;