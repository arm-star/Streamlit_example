from definitions import *


def remove_missing_values(df, col_name):
    """
    Returnes data frmae where all lines are removed checked by column if missing values exist
    :param df:
    :param col_name:
    :return:  data frame
    """

    df_clean = df.copy()

    # Remove missing values
    # Total missing values for each feature
    print('\nTotal missing values for each feature:', df_clean.isnull().sum())
    st.write('\nTotal missing values for each feature:', df_clean.isnull().sum())

    # Any missing values?
    print('\nAny missing values? ::', df_clean.isnull().values.any())
    st.write('\nAny missing values? ::', df_clean.isnull().values.any())
    df_clean = df_clean[df_clean[col_name].notna()]
    print('\nafter cleaning:', df_clean.isnull().sum())
    st.write('\nafter cleaning:', df_clean.isnull().sum())

    # Any missing values?
    print('\nAny missing values? ::', df_clean.isnull().values.any())
    st.write('\nAny missing values? ::', df_clean.isnull().values.any())
    df_clean = df_clean.reset_index(drop=True)

    return df_clean


# Future Engineering
def feature_engin_lightgbm(df, date_datetime_format):
    """
    create new datetime column
    :param df:
    :return:df
    """

    df['year'] = df[date_datetime_format].dt.year.astype('category')
    df['month'] = df[date_datetime_format].dt.month.astype('category')
    df['dayofweek_name'] = df[date_datetime_format].dt.day_name().astype('category')
    df['weekofyear'] = df[date_datetime_format].dt.weekofyear.astype('category')
    #df['quarter'] = df['timestamp'].dt.quarter.astype('category')

    ddf = df.copy()
    print(ddf.shape)
    return ddf
