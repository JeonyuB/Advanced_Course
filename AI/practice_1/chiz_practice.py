import numpy as np
from scipy.stats import chi2_contingency


if __name__ == "__main__":
    observed_data=np.array([[120, 380],
                   [180, 320]]) #전처리한 데이터(보통은 pandas 로 함)

    # 4개 튜플 = chi2: 카이제곱 통계량, p: p-value(0.05보타 작으면 상관관계 있음), dof: 자유도, expected: 기대빈도표 (배열)
    chi2, p_value,_,_ = chi2_contingency(observed_data)
    print(p_value)

