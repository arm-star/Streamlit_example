from definitions import *
from utils.visualizer_stramlit import *
from utils.preprocessing import *
from utils.model_engine import *

st.markdown("## Streamlit example using LightGBM XRP!")

#### Importing data
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    st.write('Thank you. That is the dataframe of your interest.')
    st.write(df_raw)
    st.write(f'Shape of te initial data = {df_raw.shape}')

try:
    # input target column and date column
    st.write('<style>div.Widget.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    target_column = st.radio("Select TARGET column, can NOT be Time", df_raw.columns)
    the_date_column = st.radio("Select DATE column", df_raw.columns)

    # Plot!
    st.plotly_chart(plotly_go_figure(df_raw[the_date_column],
                                     df_raw[target_column],
                                     the_date_column,
                                     target_column),
                    se_container_width=True)

    #### Make time, target columns
    target_duplicate = target_column+'_copy'
    date_datetime_format = 'timestamp'
    the_date_duplicate = date_datetime_format+'_copy'

    df = df_raw.copy()
    df[date_datetime_format] = pd.to_datetime(df[the_date_column])      # unix time to datetime
    df = df.sort_values(by=date_datetime_format, ascending=True)        # sort values by date

    #### Cut time window
    st.write('Select START and END dates ')
    day_start = '2017-6-1'
    day_end = '2020-12-31'
    day_start = st.text_input("day_start", day_start)
    day_end = st.text_input("day_end", day_end)
    mask = (df[date_datetime_format] > day_start) & (df[date_datetime_format] <= day_end)
    df = df.loc[mask].reset_index(drop=True)
    df.index = df[date_datetime_format] # set the index to time
    st.write(f'Shape of initial data {df.shape}, \n data range was cutted to: start_date={day_start}  end_date={day_end}')

    #### use visualizer to plot some info about the data
    vis_cor_heatmap(df)
    vis_group_by(df)

    #### check if missing values exist in target column and remove
    st.write('Check if missing values exist in target column and remove')
    df_clean = remove_missing_values(df, col_name=target_column)

    # duplicate target column and timestamp
    df_clean[target_duplicate] = df_clean[target_column]
    df_clean[the_date_duplicate] = df_clean[date_datetime_format]

    df_clean[date_datetime_format] = df_clean[date_datetime_format].shift(-1)
    df_clean['time_diff'] = df_clean[date_datetime_format]-df_clean[the_date_duplicate]

    # drop not necessary columns
    keep_columns_list = [target_column, date_datetime_format, target_duplicate]
    df_clean = df_clean.filter(keep_columns_list)

    # Data Preparation
    st.header('Data Preparation')

    # Shift by one hour / one time unit
    df_shift = df_clean.copy()
    df_shift[target_column] = df_shift[target_column].shift(-1)

    # Create data for algorithm
    df_data = df_shift.iloc[:-2]
    df_data = feature_engin_lightgbm(df_data, date_datetime_format) # Future Engineering

    X = df_data.drop([target_column, date_datetime_format], axis=1)
    y_0 = df_data[target_column]

    # Split Train/test sets
    st.header('Train/Test split')
    X_train, X_test, y_train, y_test = train_test_split(X, y_0, test_size=(st.number_input('test_size (%)', 20))/100,
                                                        random_state=st.number_input('random_state', 42),
                                                        shuffle=False)

    st.write(f'\nX_train {X_train.shape}, \ny_train {y_train.shape}, \nX_test {X_test.shape}, \ny_test {y_test.shape}' )


    Next_N_Points = st.number_input('Predict for the Next N Points', 14)
except:
    st.write('Please upload a csv file')

st.header('Starting the LightGBM_model')
if st.button('Start LightGBM and predict Test Set'):
    scores, model = LightGBM_model(X_train, y_train)

    st.write('cross_val_scores', scores)
    st.write('model', model)

    predictions_testset = cross_val_predict(model, X_test, y_test, cv=5)
    model_metrics(y_test, predictions_testset, 'LGBMRegressor')


    # Plot!
    st.plotly_chart(plotly_go_two_figure(df_raw[the_date_column][-len(y_test):],
                                         y_test.values.flatten(),
                                         predictions_testset.flatten(),
                                         'Time',
                                         target_column,
                                         'Actual',
                                         'Predicted',
                                         title_name='Prediction vs Test data'),
                    use_container_width=True)

    st.header('Feature Importance')

    # feature_importances
    model.fit(X_test, y_test)

    feature_imp = pd.DataFrame(sorted(zip(model.feature_importances_, X.columns)), columns=['Value','Feature'])
    data = feature_imp.sort_values(by="Value", ascending=False)
    st.write(data)
    st.write(alt.Chart(data).mark_bar().encode(
        x=alt.X('Feature', sort=None),
        y='Value',
    ))

    # Prediction for the next hour / nex time point: LightGBM
    st.header('Prediction for TODAY')
    model_h = model.fit(X, y_0)
    df_single = df_shift.copy()
    df_single = feature_engin_lightgbm(df_single, date_datetime_format)
    st.write(df_single.timestamp[-2:-1])
    df_single = df_single.iloc[-2:-1]
    X_single = df_single.drop([target_column, date_datetime_format], axis=1)
    y_0_single = df_single[target_column]#.values.reshape(-1,1)

    try:
        st.write(f'\n (TRUE VALUE = { y_0_single.values[0]} )\n (prediction = {(model_h.predict(X_single)[0])} )')
    except:
        st.write('Error Happened')


    # Prediction for the next hour: LightGBM
    st.header('Prediction for TOMORROW / next time point')

    df_single = df_shift.copy()
    df_single.timestamp[-1:] = df_single.timestamp[-2:-1]+timedelta(days=1)
    st.write(df_single.timestamp[-1:])
    df_single = feature_engin_lightgbm(df_single, date_datetime_format)
    df_single = df_single.iloc[-1:]
    X_single = df_single.drop([target_column, date_datetime_format], axis=1)
    y_0_single = df_single[target_column]#.values.reshape(-1,1)

    try:
        st.write(f'\n (TRUE VALUE = { y_0_single.values[0]} )\n (prediction = {(model_h.predict(X_single)[0])} )')
    except:
        st.write('Error Happened')

#if st.button('Prediction for the next N points'):
    # Prediction for the next N points: LightGBM
    st.header('Prediction for Next N points')
    model_h = model.fit(X, y_0)
    df_single = df_shift.copy()
    df_single.timestamp[-1:] = df_single.timestamp[-2:-1] + timedelta(days=1)
    df_single = df_single.iloc[-1:]

    n_po = []
    n_ti = []
    N_POINT=[]

    for n_points in range(Next_N_Points):
        print(n_points)

        # df_single.timestamp[-1:] = df_single.timestamp[-2:-1] + timedelta(days=1+n_points)

        df_single = feature_engin_lightgbm(df_single, date_datetime_format)
        X_single = df_single.drop([target_column, date_datetime_format], axis=1)
        y_0_single = df_single[target_column]  # .values.reshape(-1,1)

        print(f'\n (TRUE VALUE = {y_0_single.values[0]} )\n (prediction = {(model_h.predict(X_single)[0])} )')
        prediction_n = model_h.predict(X_single)[0]

        n_po.append(prediction_n)
        n_ti.append(df_single.timestamp[-1:])
        N_POINT.append(n_points)
        print(n_po)
        print(df_single.timestamp[-1:])



        df_single.timestamp[-1:] = df_single.timestamp[-1:] + timedelta(days=1 + n_points)
        df_single[target_duplicate][-1:] = prediction_n

    # Plot!
    st.plotly_chart(plotly_go_figure(N_POINT,
                                     n_po,
                                     'Next N points',
                                     target_column),
                    use_container_width=True)


st.write('END')