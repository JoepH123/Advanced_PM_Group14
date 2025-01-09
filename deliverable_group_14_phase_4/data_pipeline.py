from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from collections import Counter


def make_prefixes(df):
    """
    For each case in the DataFrame, create a 'prefix' column that is a growing list of events,
    and a 'total_time' column that represents the elapsed time in seconds since the first event
    in that case.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame must contain at least columns: ['case', 'event', 'time'].

    Returns
    -------
    pd.DataFrame
        A DataFrame with 'prefix' (list of events so far) and 'total_time' (float - total seconds from first event).
    """
    df = df.copy()
    df['prefix'] = None
    df['total_time_prefix'] = None
    df['call time'] = None

    grouped = df.groupby('case')

    # Add a progress bar to show progress through the cases
    for case_id, group in tqdm(grouped, desc="Creating prefixes", total=len(grouped)):
        prefix_list = []
        first_time = None
        first_time_call = None
        cumulative_time = 0
        for idx, row in group.iterrows():
            prefix_list.append(row['event'])
            if first_time is None:
                first_time = row['time']
            total_seconds = (row['time'] - first_time).total_seconds()
            df.at[idx, 'prefix'] = prefix_list.copy()
            df.at[idx, 'total_time_prefix'] = total_seconds

            # Compute the call time between lifecycle transitions 'start-suspend' and 'resume-suspend'
            if (row['event'] == 'W_Call after offers') or (row['event'] == 'W_Call incomplete files'):
                if row['lifecycle:transition'] == 'start':
                    first_time_call = row['time']
                    prev_transition = 'start'
                    continue

                if (row['lifecycle:transition'] == 'suspend') and (prev_transition == 'start'):
                    call_time = (row['time'] - first_time_call).total_seconds()
                    if cumulative_time == 0:
                        df.at[idx, 'call time'] = call_time
                        cumulative_time += call_time
                    else:
                        cumulative_time += call_time
                        df.at[idx, 'call time'] = cumulative_time
                    prev_transition = 'suspend'
                    continue

                if (row['lifecycle:transition'] == 'resume'):
                    first_time_call = row['time']
                    prev_transition = 'resume'
                    continue

                if (row['lifecycle:transition'] == 'suspend') and (prev_transition == 'resume'):
                    call_time = (row['time'] - first_time_call).total_seconds()
                    if cumulative_time == 0:
                        df.at[idx, 'call time'] = call_time
                        cumulative_time += call_time
                    else:
                        cumulative_time += call_time
                        df.at[idx, 'call time'] = cumulative_time
                    prev_transition = 'suspend'
                    continue


    # Convert prefixes to lists (if necessary)
    df['prefix'] = df['prefix'].apply(lambda x: list(x) if isinstance(x, list) else x)

    # Drop last row as per original code
    df = df[:-1]

    return df


def add_case_traces_and_cancel_info(df, event_of_interest):
    """
    Adds columns related to case traces and A_Cancelled occurrences.
    This function:
    - Computes prefix length.
    - Computes full case traces.
    - Maps case traces to df.
    - Checks if 'A_Cancelled' has occurred in the prefix and in the entire case.
    """
    df = df.copy()
    print("Computing prefix lengths and case traces...")

    df['prefix_length'] = df['prefix'].apply(len)
    case_traces = df.groupby('case')['event'].apply(list)
    df['case_trace'] = df['case'].map(case_traces)

    print("Checking for A_Cancelled occurrences...")
    df['event_of_interest_is_in_prefix'] = df['prefix'].apply(lambda x: x.count(event_of_interest))
    df['event_of_interest_occured'] = df['case_trace'].apply(lambda x: x.count(event_of_interest))
    df = df[(df['event_of_interest_is_in_prefix'] != 1)] # Remove rows where A_Cancelled has already happened

    return df


def index_encoding(df):
    """
    Expand the prefix column into multiple columns (one for each event in the prefix).
    """
    print("Performing index encoding...")
    df = df.copy().reset_index(drop=True)
    max_length = df['prefix'].apply(len).max()
    event_cols = [f'event_{i+1}' for i in range(max_length)]

    expanded_prefix = pd.DataFrame(df['prefix'].tolist(), columns=event_cols, index=df.index)
    df = pd.concat([df, expanded_prefix], axis=1)

    return df


def frequency_encoding(df, prefix_col='prefix'):
    """
    Perform frequency encoding within the 'prefix' column.
    Each unique event becomes a column, and the values represent the frequency
    of that event occurring in the prefix.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the 'prefix' and 'event' columns.
        prefix_col (str): Column containing the prefix data (lists of events).
        event_col (str): Column containing all unique events for reference.

    Returns:
        pd.DataFrame: DataFrame with new frequency-encoded columns.
    """
    print("Performing frequency encoding...")
    df = df.copy()
    # Extract all unique events
    unique_events = df["event"].unique()
    # Initialize columns for each unique event and calculate frequency in each prefix
    for event in unique_events:
        df[f'{event}_freq'] = df[prefix_col].apply(lambda prefix: prefix.count(event))

    return df


def frequency_encoding_opt(df, prefix_col='prefix', event_col='event'):
    """
    Create the frequency encoding for the log
    """
    df = df.copy()
    unique_events = df[event_col].unique()
    def count_events(prefix):
        event_counts = Counter(prefix)
        return [event_counts.get(event, 0) for event in unique_events]
    frequency_matrix = df[prefix_col].apply(count_events).tolist()
    freq_df = pd.DataFrame(frequency_matrix, columns=[f'freq_{event}' for event in unique_events])
    df = pd.concat([df, freq_df], axis=1)
    return df


def create_final_datasets(df, pre_offer = True):
    """
    Create final X (features) and Y (target) datasets.
    """
    print("Creating final X and Y datasets...")
    df = df.copy()

    # Perform train-test split
    df_train, df_test = split_data(df)

    # One-hot encode extra trace-specific categorical columns of interest
    categorical_columns = ['case:LoanGoal', 'case:ApplicationType']  # example categorical columns 'lifecycle:transition'
    df_train, df_test = one_hot_encode_columns_fit_transform(
        df_train, df_test, categorical_columns, drop_original=True, sparse=False, handle_unknown='ignore')

    # Remove unnecessary columns -> Change depending on new feature choices
    if pre_offer:
        remove_these_cols = [
            'Action', 'org:resource', 'EventOrigin', 'EventID',
            'prefix', 'case_trace', 'event_of_interest_is_in_prefix',
            'case', 'event', 'time', 'MonthlyCost', 'Selected', 'OfferID',
            'FirstWithdrawalAmount', 'Accepted', 'CreditScore', 'NumberOfTerms',
            'OfferedAmount', 'post_offer', 'lifecycle:transition'
            # 'case:LoanGoal', 'case:ApplicationType', 'case:RequestedAmount'
        ]
    else:
        remove_these_cols = [
            'Action', 'org:resource', 'EventOrigin', 'EventID',
            'prefix', 'case_trace', 'event_of_interest_is_in_prefix',
            'case', 'event', 'time', 'Selected', 'OfferID', 'Accepted', 'post_offer', 'lifecycle:transition'
            # 'case:LoanGoal', 'case:ApplicationType', 'case:RequestedAmount'
        ]
    df_train = df_train.drop(remove_these_cols, axis=1)
    df_test = df_test.drop(remove_these_cols, axis=1)

    df_train = convert_int64_to_int8_or_int16(df_train)
    df_test = convert_int64_to_int8_or_int16(df_test)
    df_train = convert_columns_to_float(df_train)
    df_test = convert_columns_to_float(df_test)

    X_train = df_train.drop(columns=['event_of_interest_occured'], axis=1,).copy()
    X_test = df_test.drop(columns=['event_of_interest_occured'], axis=1,).copy()

    y_train = df_train[['event_of_interest_occured']].copy()
    y_test = df_test[['event_of_interest_occured']].copy()

    return X_train, y_train, X_test, y_test


def split_data(df, train_proportion=0.8):
    """
    Create a train-test split with the given train proportion. Furthermore, we want to filter
    the train set to not contain traces which have time overlap with the traces in the test set
    """
    print("Doing the train-test split...")
    # Split dataset into train test with "train_proportion"
    lst_case = list(df['case'].unique())
    train_index = round(len(lst_case)*train_proportion)
    df_train_unfiltered = df[df['case'].isin(lst_case[:train_index])]
    df_test = df[df['case'].isin(lst_case[train_index:])]

    # Make sure that there is no time overlap between the train and test data
    max_time_train = df_test['time'].min()
    # Get the last event (row) for each case
    last_rows = df_train_unfiltered.groupby('case', as_index=False).tail(1)
    # Identify cases where the last event time is greater than max_time_train
    cases_to_exclude = last_rows.loc[last_rows['time'] > max_time_train, 'case']
    # Exclude these cases and reset index
    df_train = df_train_unfiltered[~df_train_unfiltered['case'].isin(cases_to_exclude)].reset_index(drop=True)

    df_train.reset_index(inplace=True, drop=True)
    df_test.reset_index(inplace=True, drop=True)

    train_percentage = (100*len(df_train.index)) / (len(df_train.index) + len(df_test.index))
    test_percentage = (100*len(df_test.index)) / (len(df_train.index) + len(df_test.index))
    print(f"After filtering train-test split {train_percentage}% train data and {test_percentage}% test data")

    return df_train, df_test


def compute_features(df):
    """
    Computes the features that we deem important for the prediction of the eventually follows relation with the
    event of interest.
    1. Binary indicating the presence of A_Validating in prefix
    2. Total time in prefix
    3. Prefix length
    4. Average time per event in prefix
    5. ...
    Feature idea: last_resource, last_..., requested_amount
    The last... features can just be obtained by keeping this column, because for each prefix automatically last value is stored.
    """
    # 1. Binary indicating the presence of A_Validating in prefix --> Is already in frequency encoding, so not further code necessary
    # df['a_validating_occurs'] = df['prefix'].apply(lambda lst: int("A_Validating" in lst) if isinstance(lst, list) else 0)
    # 2. This total prefix time already exists -> no further code necessary here
    # 3. This prefix length already exists -> no further code necessary here
    # 4. Average time per event in prefix must be computed by dividing total prefix time by number of events in prefix
    df["prefix_average_time"] = df["total_time_prefix"] / df["prefix_length"]
    # 5. Add offer details: extract information of O_Create Offer, and extend it after each occurrence of this event
    df['OfferedAmount'] = df.groupby('case')['OfferedAmount'].ffill()
    df['FirstWithdrawalAmount'] = df.groupby('case')['FirstWithdrawalAmount'].ffill()
    df['NumberOfTerms'] = df.groupby('case')['NumberOfTerms'].ffill()
    df['MonthlyCost'] = df.groupby('case')['MonthlyCost'].ffill()
    df['CreditScore'] = df.groupby('case')['CreditScore'].ffill()
    # Do we also use Selected and Accepted offer details?
    # 6. Difference percentage requested and offered loan amounts
    # df["loan_diff_req_and_off_perc"] = np.where(
    #     df["case:RequestedAmount"] != 0,
    #     (df["OfferedAmount"] - df["case:RequestedAmount"]) / df["case:RequestedAmount"],
    #     0
    # )
    # 7. Number of Offers Created in the prefix, is already in the frequency encoding, so no futher code necessary
    df['call time'] = df.groupby('case')['call time'].ffill()
    df['call time'].fillna(0, inplace=True) #change None values to 0
    return df


def convert_int64_to_int8_or_int16(dataframe):
    int64_cols = dataframe.select_dtypes(include=['int64']).columns
    for col in int64_cols:
        col_min = dataframe[col].min()
        col_max = dataframe[col].max()
        if col_min >= -128 and col_max <= 127:
            dataframe[col] = dataframe[col].astype('int8')
            print(f"Converted column '{col}' to int8.")
        elif col_min >= -32768 and col_max <= 32767:
            dataframe[col] = dataframe[col].astype('int16')
            print(f"Converted column '{col}' to int16.")
        else:
            print(f"Skipped column '{col}': no big downcast was done int64 was kept.")
    return dataframe


def convert_columns_to_float(dataframe):
    dataframe['total_time_prefix'] = dataframe['total_time_prefix'].astype(float)
    dataframe['prefix_average_time'] = dataframe['prefix_average_time'].astype(float)
    return dataframe


def offer_bucketing(df):
    df['post_offer'] = df['prefix'].apply(lambda x: 'O_Create Offer' in x)
    df_pre_offer = df[df.post_offer == False]
    df_post_offer = df[df.post_offer == True]
    return df_pre_offer, df_post_offer


def one_hot_encode_columns_fit_transform(train_df, test_df, categorical_columns, drop_original=True, sparse=False, handle_unknown='ignore'):
    """..."""
    encoder = OneHotEncoder(sparse_output=sparse, drop=None, handle_unknown=handle_unknown)
    encoder.fit(train_df[categorical_columns])
    train_encoded = encoder.transform(train_df[categorical_columns])
    test_encoded = encoder.transform(test_df[categorical_columns])
    if hasattr(encoder, 'get_feature_names_out'):
        feature_names = encoder.get_feature_names_out(categorical_columns)
    else:
        feature_names = encoder.get_feature_names(categorical_columns)
    if sparse:
        train_encoded_df = pd.DataFrame.sparse.from_spmatrix(train_encoded, columns=feature_names, index=train_df.index)
        test_encoded_df = pd.DataFrame.sparse.from_spmatrix(test_encoded, columns=feature_names, index=test_df.index)
    else:
        train_encoded_df = pd.DataFrame(train_encoded, columns=feature_names, index=train_df.index)
        test_encoded_df = pd.DataFrame(test_encoded, columns=feature_names, index=test_df.index)
    train_df = pd.concat([train_df, train_encoded_df], axis=1)
    test_df = pd.concat([test_df, test_encoded_df], axis=1)
    if drop_original:
        train_df = train_df.drop(columns=categorical_columns)
        test_df = test_df.drop(columns=categorical_columns)
    return train_df, test_df


def pipeline(df, event_of_interest):
    """
    Full pipeline:
    1. Create prefixes.
    2. Add case trace and cancel info.
    3. Compute all necessary features for prediction task
    4. Perform feature encoding.
    5. Prepare final X and Y datasets.
    """
    print("Starting pipeline...")
    df = make_prefixes(df)
    df = add_case_traces_and_cancel_info(df, event_of_interest)
    df = compute_features(df)
    # df = index_encoding(df)
    df = frequency_encoding(df)
    df_pre_offer, df_post_offer = offer_bucketing(df)
    X_train_pre, y_train_pre, X_test_pre, y_test_pre = create_final_datasets(df_pre_offer, pre_offer=True)
    X_train_post, y_train_post, X_test_post, y_test_post = create_final_datasets(df_post_offer, pre_offer=False)
    X_train_total, y_train_total, X_test_total, y_test_total = create_final_datasets(df, pre_offer=True)
    print("Pipeline completed.")

    return X_train_pre, y_train_pre, X_test_pre, y_test_pre, X_train_post, y_train_post, X_test_post, y_test_post, X_train_total, y_train_total, X_test_total, y_test_total

def create_final_datasets_single_case(df, pre_offer = True):
    """
    Create final X (features) and Y (target) datasets.
    """
    print("Creating final X and Y datasets...")
    df_train = df.copy()
    df_test = df.copy()

    # One-hot encode extra trace-specific categorical columns of interest
    categorical_columns = ['case:LoanGoal', 'case:ApplicationType']  # example categorical columns 'lifecycle:transition'
    df_train, df_test = one_hot_encode_columns_fit_transform(
        df_train, df_test, categorical_columns, drop_original=True, sparse=False, handle_unknown='ignore')

    # Remove unnecessary columns -> Change depending on new feature choices
    if pre_offer:
        remove_these_cols = [
            'Action', 'org:resource', 'EventOrigin', 'EventID',
            'prefix', 'case_trace', 'event_of_interest_is_in_prefix',
            'event', 'time', 'MonthlyCost', 'Selected', 'OfferID',
            'FirstWithdrawalAmount', 'Accepted', 'CreditScore', 'NumberOfTerms',
            'OfferedAmount', 'post_offer', 'lifecycle:transition'
            # 'case:LoanGoal', 'case:ApplicationType', 'case:RequestedAmount'
        ]

    else:
        remove_these_cols = [
            'Action', 'org:resource', 'EventOrigin', 'EventID',
            'prefix', 'case_trace', 'event_of_interest_is_in_prefix',
            'event', 'time', 'Selected', 'OfferID', 'Accepted', 'post_offer', 'lifecycle:transition'
            # 'case:LoanGoal', 'case:ApplicationType', 'case:RequestedAmount'
        ]
    df_train = df_train.drop(remove_these_cols, axis=1)
    df_train = convert_int64_to_int8_or_int16(df_train)
    df_train = convert_columns_to_float(df_train)

    return df_train


def begin_pipeline_single_case(df, event_of_interest):
    """
    Full pipeline:
    1. Create prefixes.
    2. Add case trace and cancel info.
    3. Compute all necessary features for prediction task
    4. Perform feature encoding.
    5. Prepare final X and Y datasets.
    """
    print("Starting pipeline...")
    df = make_prefixes(df)
    df = add_case_traces_and_cancel_info(df, event_of_interest)
    df = compute_features(df)
    df = frequency_encoding(df)
    df_pre_offer, df_post_offer = offer_bucketing(df)
    pre = create_final_datasets_single_case(df_pre_offer, pre_offer=True)
    post = create_final_datasets_single_case(df_post_offer, pre_offer=False)
    print("Pipeline completed.")

    return pre, post

def complete_pipeline_single_case(pre, post, case):
    pre = pre[pre['case']==case]
    X_pre = pre.drop(columns=['event_of_interest_occured', 'case'], axis=1,).copy()
    y_pre = pre[['event_of_interest_occured']].copy()
    post = post[post['case']==case]
    X_post = post.drop(columns=['event_of_interest_occured', 'case'], axis=1,).copy()
    y_post = pre[['event_of_interest_occured']].copy()

    print("Pipeline completed.")

    return X_pre, y_pre, X_post, y_post