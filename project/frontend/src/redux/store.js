import {configureStore} from "@reduxjs/toolkit";
import employees from "./emp/employeeSlice.js";

const store = configureStore({
    reducer: {employees}
})

export default store;