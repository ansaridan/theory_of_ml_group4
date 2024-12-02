
import pandas as pd
import numpy as np
import datetime as dt

import constants as c

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression, LinearRegression

from pathlib import Path
import logging
log = logging.getLogger("modeling")
log.setLevel(logging.INFO)
handler = logging.StreamHandler()  # This will print to the console
handler.setLevel(logging.INFO)
log.addHandler(handler)

feature_base_cols = [
    # origin
    "OriginAirportShortName",
    "OriginDivision",
    # destination
    "DestAirportShortName",
    "DestDivision",
    # flight features
    "Airline_Name",
    "Distance",
    "ScheduledDurationMinutes",
    # time features
    "FlightDate",
    "CRSDepTime",
]

DATA_PATH = Path() / "data"
AIRLINE_CODES_FILENAME = "airline_codes_map.csv"
AIRPORT_CODES_FILENAME = "airport_id_map.csv"

class DateTimeFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, flight_date_col, dep_time_col, derived_columns=["is_weekend", "month", "day_of_week", "hour_of_day"]):
        self.flight_date_col = flight_date_col
        self.dep_time_col = dep_time_col
        self.derived_columns = derived_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Ensure FlightDate and CRSDepTime are in appropriate datetime formats
        X[self.flight_date_col] = pd.to_datetime(X[self.flight_date_col], errors='coerce')
        X[self.dep_time_col] = pd.to_datetime(X[self.dep_time_col], format='%H:%M', errors='coerce')

        # Extract new features
        if "is_weekend" in self.derived_columns:
            X["is_weekend"] = X[self.flight_date_col].dt.dayofweek.isin([5, 6])
        if "month" in self.derived_columns:
            X["month"] = X[self.flight_date_col].dt.month.astype(str)
        if "day_of_week" in self.derived_columns:
            X["day_of_week"] = X[self.flight_date_col].dt.dayofweek.astype(str) + "_" + X[self.flight_date_col].dt.day_name()
        if "hour_of_day" in self.derived_columns:
            X["hour_of_day"] = X[self.dep_time_col].dt.hour.astype(str)

        # Drop the original datetime columns
        X.drop(columns=[self.flight_date_col, self.dep_time_col], inplace=True)
        return X


# Custom transformer for top-k preprocessing
class TopKPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, columns, max_n_categories=10):
        self.columns = columns
        self.max_n_categories = max_n_categories
        self.top_categories = {}

    def fit(self, X, y=None):
        # Store top-k categories for each column
        for col in self.columns:
            self.top_categories[col] = X[col].value_counts().nlargest(self.max_n_categories).index
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col] = X[col].where(X[col].isin(self.top_categories[col]), "Other")
        return X
    
class AirportMetadataPreprocessor():
    def __init__(
            self,
            added_airport_feature_cols = ["Division"]
        ):
        self.added_airport_feature_cols = added_airport_feature_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # join in airport names by code
        # https://www.transtats.bts.gov/FieldInfo.asp?Svryq_Qr5p=b4vtv0%FDNv42146%FP%FDNv42146%FDVQ.%FDN0%FDvqr06vsvpn6v10%FD07zor4%FDn55vt0rq%FDoB%FDhf%FDQbg%FD61%FDvqr06vsB%FDn%FD70v37r%FDnv42146.%FD%FDh5r%FD6uv5%FDsvryq%FDs14%FDnv42146%FDn0nyB5v5%FDnp4155%FDn%FD4n0tr%FD1s%FDBrn45%FDorpn75r%FDn0%FDnv42146%FDpn0%FDpun0tr%FDv65%FDnv42146%FDp1qr%FDn0q%FDnv42146%FDp1qr5%FDpn0%FDor%FD4r75rq.&Svryq_gB2r=a7z&Y11x72_gnoyr=Y_NVecbeg_VQ&gnoyr_VQ=FMF&flf_gnoyr_anzr=g_gEDD_ZNeXRg_NYY_PNeeVRe&fB5_Svryq_anzr=beVTVa_NVecbeg_VQ
        # NOTE: regions are defined from this CSV https://github.com/cphalpert/census-regions/blob/master/us%20census%20bureau%20regions%20and%20divisions.csv
        airport_id_map = pd.read_csv(DATA_PATH / AIRPORT_CODES_FILENAME)
        airport_id_map = airport_id_map[["Code"] + self.added_airport_feature_cols]
        X = X.join(airport_id_map.set_index("Code").rename(columns={col:f"Origin{col.replace("_", "")}" for col in airport_id_map.columns if col != "Code"}), on="OriginAirportID")
        X = X.join(airport_id_map.set_index("Code").rename(columns={col:f"Dest{col.replace("_", "")}" for col in airport_id_map.columns if col != "Code"}), on="DestAirportID")

        return X

# Function to get feature names after transformation
def get_feature_names(column_transformer, input_features):
    feature_names = []
    for name, transformer, columns in column_transformer.transformers_:
        if name == 'remainder' and transformer == 'passthrough':
            feature_names.extend(columns)  # for passthrough columns, retain original names
        elif isinstance(transformer, OneHotEncoder):
            feature_names.extend(transformer.get_feature_names_out(columns))
        elif isinstance(transformer, StandardScaler):
            feature_names.extend(columns)  # StandardScaler retains original column names
    return feature_names

class FlightPerformanceModel():
    def __init__(
            self,
            targets = {
                c.CANCELLED_COL: {
                    "type": "classification",
                    "estimator": LogisticRegression,
                    "params": {}
                },
                c.DELAYED_COL: {
                    "type": "classification",
                    "estimator": LogisticRegression,
                    "params": {}
                },
                c.DELAY_DURATION_COL: {
                    "type": "regression",
                    "estimator": LinearRegression,
                    "params": {}
                },
            },
            one_hot_features = [
                "OriginAirportShortName", "DestAirportShortName",
                "OriginDivision", "DestDivision", "Airline_Name",
            ],
            top_k_features = [
                "OriginAirportShortName", "DestAirportShortName",
            ],
            numerical_features = ["Distance", "ScheduledDurationMinutes"],
            datetime_features = ["is_weekend", "month", "day_of_week", "hour_of_day"],
            flight_date_col="FlightDate",
            dep_time_col="CRSDepTime"
        ):
        self.targets = targets
        self.estimators = {}
        self.features = {
            "one_hot": one_hot_features,
            "top_k": top_k_features,
            "numerical": numerical_features,
            "datetime": datetime_features,
        }
        self.flight_date_col = flight_date_col
        self.dep_time_col = dep_time_col
        self.target_cols = [k for k, v in self.targets.items()]

    def fit(self, df):
        start_time = dt.datetime.now()

        self._fit_transformer(df)
        X_train = self._transform(df)
        # remove rows with NaN
        rows_with_nan_feats = np.isnan(X_train).any(axis=1)
        count_nan_rows = rows_with_nan_feats.sum()
        log.info(f"Number of train rows with NaN: {count_nan_rows}")

        X_train = X_train[~rows_with_nan_feats]
        y_train = df[self.target_cols]
        y_train = y_train[~rows_with_nan_feats].fillna(0)

        """Fit models for each target column."""
        for target_col, target_metadata in self.targets.items():
            if target_metadata["type"] == "classification":
                log.info(f"{dt.datetime.now():%H:%M:%S}:Training model for target {target_col}")
                estimator = target_metadata["estimator"]
                estimator_params = target_metadata["params"]
                self.estimators[target_col] = estimator(**estimator_params).fit(X_train, y_train[target_col])
                log.info(f"{dt.datetime.now():%H:%M:%S}:Model for target {target_col} trained.")
        
        if c.DELAY_DURATION_COL in self.target_cols:
            log.info(f"{dt.datetime.now():%H:%M:%S}:Training model for target {c.DELAY_DURATION_COL}")
            target_col, target_metadata = c.DELAY_DURATION_COL, self.targets[c.DELAY_DURATION_COL]
            estimator = target_metadata["estimator"]
            estimator_params = target_metadata["params"]
            self.estimators[c.DELAY_DURATION_COL] = estimator(**estimator_params).fit(
                X_train[(y_train[c.DELAYED_COL] == 1.0).values, :],
                y_train[y_train[c.DELAYED_COL] == 1.0][c.DELAY_DURATION_COL]
            )
            log.info(f"{dt.datetime.now():%H:%M:%S}:Model for target {c.DELAY_DURATION_COL} trained.")
        
        end_time = dt.datetime.now()
        self.train_time = end_time - start_time
    
    def _fit_transformer(self, df):
        # Define columns for transformers
        one_hot_columns = self.features["one_hot"]
        top_k_columns = self.features["top_k"]
        numerical_columns = self.features["numerical"]
        datetime_columns = self.features["datetime"]
        feature_base_cols = self.features["one_hot"] + self.features["numerical"] + [self.flight_date_col, self.dep_time_col]

        # Define the pipeline
        pipeline = Pipeline([
            ('datetime_features', DateTimeFeatureExtractor(
                flight_date_col="FlightDate", dep_time_col="CRSDepTime",
                derived_columns=datetime_columns
            )),
            ('top_k_preprocess', TopKPreprocessor(columns=top_k_columns, max_n_categories=10)),
            ('preprocess', ColumnTransformer([
                ('one_hot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), 
                    one_hot_columns + datetime_columns
                ),
                ('scaler', StandardScaler(), numerical_columns)
            ], remainder='passthrough')),
            ('final_scaler', StandardScaler())  # Standard scaling applied to all columns after preprocessing
        ])

        pipeline.fit(df[feature_base_cols])

        # Get column names after transformation
        preprocessor = pipeline.named_steps['preprocess']

        self.column_names = get_feature_names(preprocessor, df[feature_base_cols].columns)
        self.feature_base_cols = feature_base_cols
        self.transformer = pipeline

    def _transform(self, df):
        return self.transformer.transform(df[self.feature_base_cols])

    def _transform_to_df(self, df):
        X_transformed = self.transformer.transform(df[self.feature_base_cols])
        X_transformed = pd.DataFrame(X_transformed, columns=self.column_names)
        return X_transformed
        

    def predict_proba(self, df, target_col):
        """Predict probabilities for a specific target column on the test set."""
        X_test = self._transform(df)
        model = self.estimators.get(target_col)
        if model is None:
            raise ValueError(f"Model for target {target_col} has not been trained.")
        return model.predict_proba(X_test)[:, 1]

    def predict(self, df, target_col):
        """Predict the target column on the test set."""
        X_test = self._transform(df)
        model = self.estimators.get(target_col)
        if model is None:
            raise ValueError(f"Model for target {target_col} has not been trained.")
        return model.predict(X_test)

    def predict_on_time(self, df):
        df_pred = pd.DataFrame()
        for target_col in [c.DELAYED_COL, c.DELAY_DURATION_COL, c.CANCELLED_COL]:
            model = self.estimators.get(target_col)
            if model is None:
                raise ValueError(f"Model for target {target_col} has not been trained.")
            if target_col == c.DELAY_DURATION_COL:
                y_pred = self.predict(df, target_col=target_col)
            else:
                y_pred = self.predict_proba(df, target_col=target_col)
            df_pred[target_col] = y_pred
        
        return df_pred