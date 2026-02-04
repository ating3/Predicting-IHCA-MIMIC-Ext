import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.tuner import Tuner
from sklearn.model_selection import train_test_split
from pytorch_forecasting.metrics import CrossEntropy
from pytorch_forecasting.data.encoders import NaNLabelEncoder
import numpy
import os

# load data
#input_vitals = args.input
input_vitals = os.getenv("BASE", "data.csv")
data = pd.read_csv(input_vitals)
data = data.astype({"diagnosis": str})
data = data.sort_values(["subject_id","stay_id","bin_hour"])
data = data.drop_duplicates(subset=["subject_id","stay_id","bin_hour"], keep="last")
# define dataset
max_encoder_length = 13
#look back window: input window size of data
max_prediction_length = 1
#min_encoder_length = 1
#look ahead window: prediction label window size

data["split"] = None
train, val= train_test_split(range(len(data)), train_size=0.8 ,shuffle=True)
data.iloc[train, data.columns.get_loc("split")] = "train"
data.iloc[val, data.columns.get_loc("split")] = "val"
train_data = data[data["split"]=="train"]
val_data = data[data["split"]=="val"]

train_set = TimeSeriesDataSet(
    train_data,
    time_idx= "bin_hour",
    target= "label",
    # weight="weight",
    group_ids=["subject_id", "stay_id"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=["admission_type", "sex", "race", "diagnosis"],
    static_reals=["age"],
    #variables that are known in the future(price of a product, etc.)
    #time_varying_known_categoricals=[ ... ],
    #time_varying_known_reals=[ ... ],
    #time_varying_unknown_categoricals=[ ... ],
    time_varying_unknown_reals=["HR", "RR", "SpO2", "SBP", "DBP", "MAP"],
    allow_missing_timesteps=True,
    #min_encoder_length=min_encoder_length
)

val_set = TimeSeriesDataSet(
    val_data,
    time_idx= "bin_hour",
    target= "label",
    # weight="weight",
    group_ids=["subject_id", "stay_id"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=["admission_type", "sex", "race", "diagnosis"],
    static_reals=["age"],
    #variables that are known in the future(price of a product, etc.)
    #time_varying_known_categoricals=[ ... ],
    #time_varying_known_reals=[ ... ],
    #time_varying_unknown_categoricals=[ ... ],
    time_varying_unknown_reals=["HR", "RR", "SpO2", "SBP", "DBP", "MAP"],
    allow_missing_timesteps=True,
    #min_encoder_length=min_encoder_length
)

# create validation and training dataset
batch_size = 32
train_dataloader = train_set.to_dataloader(train=True, batch_size=batch_size, num_workers=2)
val_dataloader = val_set.to_dataloader(train=False, batch_size=batch_size, num_workers=2)

early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=1, verbose=False, mode="min")
lr_logger = LearningRateMonitor()
trainer = L.Trainer(
    max_epochs=2,
    accelerator="auto",
    gradient_clip_val=0.1,
    limit_train_batches=30,
    callbacks=[lr_logger, early_stop_callback]
)
    
tft = TemporalFusionTransformer.from_dataset(
    train_set,
    learning_rate=0.03,
    hidden_size=32,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=16,
    output_size=2,
    loss=CrossEntropy(),
    log_interval=2,
    reduce_on_plateau_patience=4
)

print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

_original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    # force weights_only=False
    if 'weights_only' in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = patched_torch_load

res = Tuner(trainer).lr_find(
tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, early_stop_threshold=1000.0, max_lr=0.3,
)

torch.load = _original_torch_load

print(f"suggested learning rate: {res.suggestion()}")
fig = res.plot(show=True, suggest=True)
fig.show()

# fit the model
trainer.fit(
    tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader,
)
