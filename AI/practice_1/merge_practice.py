import pandas as pd

#테이블 병합(merge)하는 작업: 분석하기 위해 만들음.

if __name__ == '__main__':
    sales_df = pd.read_excel("data/Sales.xlsx", sheet_name="Sheet1")#대소문자 틀리면 오류뜸.
    details_df = pd.read_excel("data/Details.xlsx", sheet_name=None)
    print(details_df.keys())
    regions_df = details_df["지역"]
    promotion_df = details_df["프로모션"]
    channel_df = details_df["채널"]
    date_df = details_df["날짜"]
    customer_df = details_df["2018년도~2022년도 주문고객"]
    product_df = details_df["제품"]
    category_df = details_df["분류"]
    product_category_df = details_df["제품분류"]
    merged_df = pd.merge(sales_df, product_df, on="제품코드", how="left")
    merged_df = pd.merge(merged_df, customer_df, on="고객코드", how="left")
    merged_df = pd.merge(merged_df, promotion_df, on="프로모션코드", how="left")
    merged_df = pd.merge(merged_df, product_category_df, on="제품분류코드", how="left")
    merged_df = pd.merge(merged_df, regions_df, on="지역코드", how="left")
    merged_df = pd.merge(merged_df, date_df, on="날짜", how="left")
    merged_df = pd.merge(merged_df, category_df, on="분류코드", how="left")
    print(merged_df.keys())

    columns = ['날짜', 'Quantity', '지역_x',
       '제품명', '색상', '원가', '단가', '고객명', '성별', '생년월일', '프로모션',
       '할인율', '제품분류명', '시도', '구군시', '지역_y', '년도', '분기',
       '월(No)', '월(영문)', '분류명']
    merged_df = merged_df[columns]
    merged_df.rename(columns={'Quantity':'수량','지역_x': '지역'}, inplace=True)
    print(merged_df.keys())

    #inplace=True없으면 rename이 안먹힘. 즉, 원래는 immutable(불변) 하다는 뜻.inplace=True를 붙여야 수정이 먹힘.

    #fancy indexing 및 projection
    print(merged_df["단가"])
    #projection
    merged_df["판매가"]=merged_df["단가"]*merged_df["수량"]*(1-merged_df["할인율"])
    #fancy indexing:조건을 걸어 원하는 값을 가져올 수 있다
    condition = merged_df['판매가']>=50000
    print(merged_df[condition])
    tmp_df = merged_df[condition]
    print(tmp_df["판매가"])