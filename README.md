# transforming-prosit
Transforming Prosit: Investigation of Transformer-Based Approach to Predict Tandem Spectrum Intensities

## Data
- To train a variable input length model, use **get_tfdatasets** from *prosit_t.data.parquet_to_tfdataset*
- To train a fixed input length model, use **get_tfdatasets_padded_filtered** from *prosit_t.data.parquet_to_tfdataset_padded_filtered*
  - This function is also responsible for filtering the data only for FTMS as a mass analyzer type 

## Models
The project introduces two types of models
- The model that takes peptide sequences with the fixed length (>=7 & <= 30)
  - The implementation for the optimal model can be found under **prosit_transformer_v2** in *prosit_t.models*

  <img src="https://github.com/Lizi3107/transforming-prosit/assets/47035093/633ca915-da2b-409c-b442-693a0c8d9af9" alt="model" width="200"/>
  
- The model that does not limit the length of the input sequence lengths
  - The implementation for the optimal model can be found under **PrositTransformerDynamicLenDropLast** in *prosit_t.models.variable_seq_length_models*

    ![dynamic_model](https://github.com/Lizi3107/transforming-prosit/assets/47035093/43ca81c8-fb04-41dc-94be-25d346ccdeef)


## Train
- Run prosit_t.wandb_agent.train for every transformer-based model training

- Run prosit_t.wandb_agent.train_prosit to train the Prosit baseline model

## Model Config
- Training pipeline should be provided with the path to the model config yaml file 
- Optimal parameters for the fixed sequence input length model can be found at transformer_v2.yaml in prosit_t.wandb_agent.model_configs
- Optimal parameters for the dynamic sequence input length model can be found at transformer_dynamic.yaml in prosit_t.wandb_agent.model_configs
