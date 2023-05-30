import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

def main():
    st.title('K-Means 클러스터링 앱')

    # 1. csv파일을 업로드 할 수 있다
    csv_file = st.file_uploader('csv 파일 업로드', type=['csv'])

    if csv_file is not None:
        # 업로드한 csv파일을 데이터프레임으로 읽고
        df = pd.read_csv(csv_file)

        st.dataframe(df) # 보여줘라
        
        st.subheader('NaN 데이터 확인')
        
        st.dataframe(df.isna().sum()) # 비어있는 데이터 보여주기
        
        st.subheader('결측한 데이터 처리 결과')

        df.dropna()
        df.reset_index(inplace = True, drop=True)

        st.dataframe(df)
        
        st.subheader('클러스터링에 사용할 컬럼 선택')
        
        selected_columns = st.multiselect('X로 사용할 컬럼을 선택하세요', df.columns)
        
        if len(selected_columns) != 0:
        
            X = df[selected_columns]
            st.dataframe( X )

        # 숫자로된 새로운 데이터프레임을 만든다
        X_new = pd.DataFrame()

        for name in X.columns:
            # print(name)

         #데이터가 문자열이면, 데이터의 종류가 몇개인지 확인하다.
            if X[ name ].dtype == object:

               if X[ name ].nunique() >= 3:
                 # 원핫 인코딩한다
                ct = ColumnTransformer( [('encoder', OneHotEncoder(), [0] )] , 
                       remainder= 'passthrough' )
      
                col_names = sorted( X[name].unique() )
      
                X_new[col_names] = ct.fit_transform( X[name].to_frame() )

            else:
                # 레이블 인코딩한다
                label_encoder = LabelEncoder()
                X_new[name] = label_encoder.fit_transform(X[name])

      # 숫자데이터일 때의 처리는 여기서
    else:
        X_new[name] = X[name]            

    st.subheader('문자열은 숫자로 바꿔줍니다')
    st.dataframe(X_new)

    st.subheader('피쳐스케일링 합니다')
    scaler = MinMaxScaler()
    X_new = scaler.fit_transform(X_new)
    st.dataframe(X_new)

    # 유저가 입력한 파일의 데이터 갯수를 세어서
    # 해당 데이터의 갯수가 10보다 작으면,
    # 데이터의 갯수까지만 WCSS를 구하고,
    # 10보다 크면, 10개로 한다

    if X_new.shape[0] < 10: 
        max_count = X_new.shape[0]
    else :
        max_count = 10
    # 데이터 억세스는 [] 대괄호
    # (8, 5) 데이터에 따라서 달라질 것

    wcss=[]
    for k in range(1, max_count+1) :
        kmeans = KMeans(n_clusters= k, random_state=5, n_init='auto') 
        kmeans.fit(X_new)
        wcss.append(kmeans.inertia_) # 맨 위에 있는 wcss리스트 안에 집어넣기 위한 코드

    x = np.arange(1, max_count+1)
    fig = plt.figure()
    plt.plot(x, wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Clusters')
    plt.ylabel('WCSS')
    st.pyplot(fig)

    st.subheader('클러스터링 갯수 선택')

    k = st.number_input('k 를 선택', 1, max_count, value=3)

    kmeans = KMeans(n_clusters=k, random_state=5, n_init='auto')

    y_pred = kmeans.fit_predict(X_new)

    df['Group'] = y_pred

    st.subheader('그루핑 정보 표시')
    st.dataframe(df)

    st.subheader('보고싶은 그룹을 선택하세요')
    group_number = st.number_input('그룹번호 선택', 0, k-1)

    st.dataframe(df.loc[df['Group'] == group_number, ])

    df.to_csv('result_csv', index=False)


if __name__ == '__main__':
    main()
