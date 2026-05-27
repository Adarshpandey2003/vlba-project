# This is an example feature definition file

from datetime import timedelta

import pandas as pd

from feast import (
    Entity,
    FeatureService,
    FeatureView,
    Field,
    FileSource,
    Project,
    PushSource,
    RequestSource,
    ValueType,
)
from feast.feature_logging import LoggingConfig
from feast.infra.offline_stores.file_source import FileLoggingDestination
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Float32, Float64, Int64, Json, Map, String, Struct


customer = Entity(name="passenger_id", value_type=ValueType.INT64, description="Passenger ID")

file_source= FileSource(
    path=r"data/predictor_data.parquet",
    timestamp_field="event_timestamp",
)


#predictor fv
predictor_fv=FeatureView(
    name="predictor_features",
    entities=[customer],
    ttl=timedelta(days=3),
    schema=[
        Field(name="age", dtype=Float32),
        Field(name="flight_distance", dtype=Float32),
        Field(name="seat_comfort", dtype=Int64),
        Field(name="departure_arrival_time_convenient", dtype=Int64),
        Field(name="food_and_drink", dtype=Int64),
        Field(name="gate_location", dtype=Int64),
        Field(name="inflight_wifi_service", dtype=Int64),
        Field(name="inflight_entertainment", dtype=Int64),
        Field(name="online_support", dtype=Int64),
        Field(name="ease_of_online_booking", dtype=Int64),
        Field(name="onboard_service", dtype=Int64),
        Field(name="leg_room_service", dtype=Int64),
        Field(name="baggage_handling", dtype=Int64),
        Field(name="checkin_service", dtype=Int64),
        Field(name="cleanliness", dtype=Int64),
        Field(name="online_boarding", dtype=Int64),
        Field(name="departure_delay", dtype=Float32),
        Field(name="arrival_delay", dtype=Float32),
        Field(name="total_service_score", dtype=Float32),
        Field(name="total_delay", dtype=Float32),
        Field(name="is_long_flight", dtype=Int64),
        Field(name="customer_type_loyal", dtype=Int64),
        Field(name="customer_type_disloyal", dtype=Int64),
        Field(name="travel_type_business", dtype=Int64),
        Field(name="travel_type_personal", dtype=Int64),
        Field(name="class_business", dtype=Int64),
        Field(name="class_eco", dtype=Int64),
        Field(name="class_eco_plus", dtype=Int64),
    ],
    source=file_source,
    online=True,
    tags={}
)


# target fv
tgt_file_source=FileSource(
    path=r"data/target_data.parquet",
    timestamp_field="event_timestamp",
)
target_fv=FeatureView(
    name="target_features",
    entities=[customer],
    ttl=timedelta(days=3),
    schema=[
        Field(name="satisfaction", dtype=Int64),
    ],
    source=tgt_file_source,
    online=True,
    tags={}
)
