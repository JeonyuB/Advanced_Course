import React from 'react';
import { Link } from 'react-router-dom';

const MainPage = () => {
    return (
        <div>
            <Link to="/product">상품</Link>
            <Link to="/sales">세일즈</Link>
            <Link to="/category">카테</Link>
        </div>
    );
};

export default MainPage;