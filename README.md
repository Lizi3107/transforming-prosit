# transforming-prosit
Transforming Prosit: Investigation of Transformer-Based Approach to Predict Tandem Spectrum Intensities

## Data
To train a variable input length model, use get_tfdatasets from prosit_t.data.parquet_to_tfdataset 
To train a fixed input length model, use get_tfdatasets_padded_filtered from prosit_t.data.parquet_to_tfdataset_padded_filtered (the data is filtered for only FTMS samples)

## Models
The implementation for the optimal model with fixed sequence input length can be found at prosit_t.models.prosit_transformer_v2
The implementation for the optimal model with dynamic sequence input length can be found at prosit_t.models.variable_seq_length_models.PrositTransformerDynamicLenDropLast

## Train
Run prosit_t.wandb_agent.train for every transformer-based model training
Run prosit_t.wandb_agent.train_prosit to train the Prosit baseline model

## Model Config
Training pipeline should be provided with the path to the model config yaml file 
Optimal parameters for the fixed sequence input length model can be found at transformer_v2.yaml in prosit_t.wandb_agent.model_configs
Optimal parameters for the dynamic sequence input length model can be found at transformer_dynamic.yaml in prosit_t.wandb_agent.model_configs
