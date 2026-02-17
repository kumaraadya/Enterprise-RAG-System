---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:2
- loss:BinaryCrossEntropyLoss
base_model: cross-encoder/ms-marco-MiniLM-L6-v2
pipeline_tag: text-ranking
library_name: sentence-transformers
---

# CrossEncoder based on cross-encoder/ms-marco-MiniLM-L6-v2

This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) using the [sentence-transformers](https://www.SBERT.net) library. It computes scores for pairs of texts, which can be used for text reranking and semantic search.

## Model Details

### Model Description
- **Model Type:** Cross Encoder
- **Base model:** [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) <!-- at revision c5ee24cb16019beea0893ab7796b1df96625c6b8 -->
- **Maximum Sequence Length:** 512 tokens
- **Number of Output Labels:** 1 label
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Documentation:** [Cross Encoder Documentation](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Cross Encoders on Hugging Face](https://huggingface.co/models?library=sentence-transformers&other=cross-encoder)

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import CrossEncoder

# Download from the 🤗 Hub
model = CrossEncoder("cross_encoder_model_id")
# Get scores for pairs of texts
pairs = [
    ['2021-12-31\n0001018724\nus-gaap:FairValueInputsLevel2Member\nus-gaap:FairValueMeasurementsRecurringMember\nus-gaap:FixedIncomeSecuritiesMember\n2021-12-31\n0001018724\nus-gaap:FairValueInputsLevel2Member\nus-', 'advertiser spending; ongoing product and policy changes; and, as it relates to paid clicks, fluctuations in search queries resulting from changes in user adoption and usage, primarily on mobile devices.\nChanges in cost-per-click and cost-per-impression are driven by a number of interrelated factors including changes in device mix, geographic mix, advertiser spending, ongoing product and policy changes, product mix, property mix, and changes in foreign currency exchange rates.\n36.\nTable of Contents\nAlphabet Inc.\nGoogle subscriptions, platforms, and devices\nGoogle subscriptions, platforms, and devices revenues increased $5.7 billion from 2023 to 2024. The growth was primarily driven by an increase in subscription revenues, largely from growth in the number of paid subscribers for YouTube services followed by Google One.\nGoogle Cloud\nGoogle Cloud revenues increased\n $10.1 billion\nfrom 2023 to 2024 primarily driven by growth in Google Cloud Platform largely from infrastructure services.\nRevenues by Geography\nThe following table presents revenues by geography as a percentage of revenues, determined based on the addresses of our customers:\n\xa0\nYear Ended December 31,\n\xa0\n2023\n2024\nUnited States\n47\xa0\n%\n49\xa0\n%\nEMEA\n30\xa0\n%\n29\xa0\n%\nAPAC\n17\xa0\n%\n16\xa0\n%\nOther Americas\n6\xa0\n%\n6\xa0\n%\nHedging gains (losses)\n0\xa0\n%\n0\xa0\n%\nFor additional information, see Note 2 of the Notes to Consolidated Financial Statements included in Item 8 of this Annual Report on Form 10-K.\nUse of Non-GAAP Constant Currency Information\nInternational revenues, which represent a significant portion of our revenues, are generally transacted in multiple currencies and\xa0therefore are affected by fluctuations in foreign currency exchange rates.\nThe effect of currency exchange rates on our business is an important factor in understanding period-to-period comparisons. We use non-GAAP constant currency revenues ("constant currency revenues") and non-GAAP percentage change in constant currency revenues ("percentage change in constant currency revenues") for financial and operational decision-making and as a means to evaluate period-to-period comparisons. We believe the presentation of results on a constant currency basis in addition to U.S. Generally Accepted Accounting Principles (GAAP) results helps improve the ability to understand our performance, because it excludes the effects of foreign currency volatility that are not indicative of our core operating results.\nConstant currency information compares results between periods as if exchange rates had remained constant period over period. We define constant currency revenues as revenues excluding the effect of foreign currency exchange rate movements'],
    ['2021-12-31\n0001018724\nus-gaap:FairValueInputsLevel2Member\nus-gaap:FairValueMeasurementsRecurringMember\nus-gaap:FixedIncomeSecuritiesMember\n2021-12-31\n0001018724\nus-gaap:FairValueInputsLevel2Member\nus-', 'Business Places Increased Strain on Our Operations\nDemand for our products and services can fluctuate significantly for many reasons, including as a result of seasonality, promotions, product launches, or unforeseeable events, such as in response to global economic conditions such as recessionary fears or rising inflation, natural or human-caused disasters (including public health crises) or extreme weather (including as a result of climate change), or geopolitical events. For example, we expect a disproportionate amount of our retail sales to occur during our fourth quarter. Our failure to stock or restock popular products in sufficient amounts such that we fail to meet customer demand could significantly affect our revenue and our future growth. When we overstock products, we may be required to take significant inventory markdowns or write-offs and incur commitment costs, which could materially reduce profitability. We regularly experience increases in our net shipping cost due to complimentary upgrades, split-shipments, and additional long-zone shipments necessary to ensure timely delivery for the holiday season. If too many customers access our websites within a short period of time due to increased demand, we may experience system interruptions that make our websites unavailable or prevent us from efficiently fulfilling orders, which may reduce the volume of goods we offer or sell and the attractiveness of our products and services. In addition, we may be unable to adequately staff our fulfillment network and customer service centers during these peak periods and delivery and other fulfillment companies and customer service co-sourcers may be unable to meet the seasonal demand. Risks described elsewhere in this Item\xa01A relating to fulfillment network optimization and inventory are magnified during periods of high demand.\nAs a result of holiday sales, as of December\xa031 of each year, our cash, cash equivalents, and marketable securities balances typically reach their highest level (other than as a result of cash flows provided by or used in investing and financing activities) because consumers primarily use credit cards in our stores and the related receivables settle quickly. Typically, there is also a corresponding increase in accounts payable as of December\xa031 due to inventory purchases and third-party seller sales. Our accounts payable balance generally declines during the first three months of the year as vendors and sellers are paid, resulting in a corresponding decline in our cash, cash equivalents, and marketable securities balances.\nWe Are Impacted by Fraudulent or Unlawful Activities of Sellers\nThe law relating to the liability of online service providers is currently unsettled. In addition, governmental agencies have in the past and could in the future require changes in the way this business is conducted'],
]
scores = model.predict(pairs)
print(scores.shape)
# (2,)

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    '2021-12-31\n0001018724\nus-gaap:FairValueInputsLevel2Member\nus-gaap:FairValueMeasurementsRecurringMember\nus-gaap:FixedIncomeSecuritiesMember\n2021-12-31\n0001018724\nus-gaap:FairValueInputsLevel2Member\nus-',
    [
        'advertiser spending; ongoing product and policy changes; and, as it relates to paid clicks, fluctuations in search queries resulting from changes in user adoption and usage, primarily on mobile devices.\nChanges in cost-per-click and cost-per-impression are driven by a number of interrelated factors including changes in device mix, geographic mix, advertiser spending, ongoing product and policy changes, product mix, property mix, and changes in foreign currency exchange rates.\n36.\nTable of Contents\nAlphabet Inc.\nGoogle subscriptions, platforms, and devices\nGoogle subscriptions, platforms, and devices revenues increased $5.7 billion from 2023 to 2024. The growth was primarily driven by an increase in subscription revenues, largely from growth in the number of paid subscribers for YouTube services followed by Google One.\nGoogle Cloud\nGoogle Cloud revenues increased\n $10.1 billion\nfrom 2023 to 2024 primarily driven by growth in Google Cloud Platform largely from infrastructure services.\nRevenues by Geography\nThe following table presents revenues by geography as a percentage of revenues, determined based on the addresses of our customers:\n\xa0\nYear Ended December 31,\n\xa0\n2023\n2024\nUnited States\n47\xa0\n%\n49\xa0\n%\nEMEA\n30\xa0\n%\n29\xa0\n%\nAPAC\n17\xa0\n%\n16\xa0\n%\nOther Americas\n6\xa0\n%\n6\xa0\n%\nHedging gains (losses)\n0\xa0\n%\n0\xa0\n%\nFor additional information, see Note 2 of the Notes to Consolidated Financial Statements included in Item 8 of this Annual Report on Form 10-K.\nUse of Non-GAAP Constant Currency Information\nInternational revenues, which represent a significant portion of our revenues, are generally transacted in multiple currencies and\xa0therefore are affected by fluctuations in foreign currency exchange rates.\nThe effect of currency exchange rates on our business is an important factor in understanding period-to-period comparisons. We use non-GAAP constant currency revenues ("constant currency revenues") and non-GAAP percentage change in constant currency revenues ("percentage change in constant currency revenues") for financial and operational decision-making and as a means to evaluate period-to-period comparisons. We believe the presentation of results on a constant currency basis in addition to U.S. Generally Accepted Accounting Principles (GAAP) results helps improve the ability to understand our performance, because it excludes the effects of foreign currency volatility that are not indicative of our core operating results.\nConstant currency information compares results between periods as if exchange rates had remained constant period over period. We define constant currency revenues as revenues excluding the effect of foreign currency exchange rate movements',
        'Business Places Increased Strain on Our Operations\nDemand for our products and services can fluctuate significantly for many reasons, including as a result of seasonality, promotions, product launches, or unforeseeable events, such as in response to global economic conditions such as recessionary fears or rising inflation, natural or human-caused disasters (including public health crises) or extreme weather (including as a result of climate change), or geopolitical events. For example, we expect a disproportionate amount of our retail sales to occur during our fourth quarter. Our failure to stock or restock popular products in sufficient amounts such that we fail to meet customer demand could significantly affect our revenue and our future growth. When we overstock products, we may be required to take significant inventory markdowns or write-offs and incur commitment costs, which could materially reduce profitability. We regularly experience increases in our net shipping cost due to complimentary upgrades, split-shipments, and additional long-zone shipments necessary to ensure timely delivery for the holiday season. If too many customers access our websites within a short period of time due to increased demand, we may experience system interruptions that make our websites unavailable or prevent us from efficiently fulfilling orders, which may reduce the volume of goods we offer or sell and the attractiveness of our products and services. In addition, we may be unable to adequately staff our fulfillment network and customer service centers during these peak periods and delivery and other fulfillment companies and customer service co-sourcers may be unable to meet the seasonal demand. Risks described elsewhere in this Item\xa01A relating to fulfillment network optimization and inventory are magnified during periods of high demand.\nAs a result of holiday sales, as of December\xa031 of each year, our cash, cash equivalents, and marketable securities balances typically reach their highest level (other than as a result of cash flows provided by or used in investing and financing activities) because consumers primarily use credit cards in our stores and the related receivables settle quickly. Typically, there is also a corresponding increase in accounts payable as of December\xa031 due to inventory purchases and third-party seller sales. Our accounts payable balance generally declines during the first three months of the year as vendors and sellers are paid, resulting in a corresponding decline in our cash, cash equivalents, and marketable securities balances.\nWe Are Impacted by Fraudulent or Unlawful Activities of Sellers\nThe law relating to the liability of online service providers is currently unsettled. In addition, governmental agencies have in the past and could in the future require changes in the way this business is conducted',
    ]
)
# [{'corpus_id': ..., 'score': ...}, {'corpus_id': ..., 'score': ...}, ...]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 2 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 2 samples:
  |         | sentence_0                                                                                       | sentence_1                                                                                          | label                                                         |
  |:--------|:-------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------|:--------------------------------------------------------------|
  | type    | string                                                                                           | string                                                                                              | float                                                         |
  | details | <ul><li>min: 200 characters</li><li>mean: 200.0 characters</li><li>max: 200 characters</li></ul> | <ul><li>min: 2676 characters</li><li>mean: 2773.0 characters</li><li>max: 2870 characters</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.5</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                    | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | label            |
  |:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>2021-12-31<br>0001018724<br>us-gaap:FairValueInputsLevel2Member<br>us-gaap:FairValueMeasurementsRecurringMember<br>us-gaap:FixedIncomeSecuritiesMember<br>2021-12-31<br>0001018724<br>us-gaap:FairValueInputsLevel2Member<br>us-</code> | <code>advertiser spending; ongoing product and policy changes; and, as it relates to paid clicks, fluctuations in search queries resulting from changes in user adoption and usage, primarily on mobile devices.<br>Changes in cost-per-click and cost-per-impression are driven by a number of interrelated factors including changes in device mix, geographic mix, advertiser spending, ongoing product and policy changes, product mix, property mix, and changes in foreign currency exchange rates.<br>36.<br>Table of Contents<br>Alphabet Inc.<br>Google subscriptions, platforms, and devices<br>Google subscriptions, platforms, and devices revenues increased $5.7 billion from 2023 to 2024. The growth was primarily driven by an increase in subscription revenues, largely from growth in the number of paid subscribers for YouTube services followed by Google One.<br>Google Cloud<br>Google Cloud revenues increased<br> $10.1 billion<br>from 2023 to 2024 primarily driven by growth in Google Cloud Platform largely from infrastructure services.<br>Re...</code> | <code>0.0</code> |
  | <code>2021-12-31<br>0001018724<br>us-gaap:FairValueInputsLevel2Member<br>us-gaap:FairValueMeasurementsRecurringMember<br>us-gaap:FixedIncomeSecuritiesMember<br>2021-12-31<br>0001018724<br>us-gaap:FairValueInputsLevel2Member<br>us-</code> | <code>Business Places Increased Strain on Our Operations<br>Demand for our products and services can fluctuate significantly for many reasons, including as a result of seasonality, promotions, product launches, or unforeseeable events, such as in response to global economic conditions such as recessionary fears or rising inflation, natural or human-caused disasters (including public health crises) or extreme weather (including as a result of climate change), or geopolitical events. For example, we expect a disproportionate amount of our retail sales to occur during our fourth quarter. Our failure to stock or restock popular products in sufficient amounts such that we fail to meet customer demand could significantly affect our revenue and our future growth. When we overstock products, we may be required to take significant inventory markdowns or write-offs and incur commitment costs, which could materially reduce profitability. We regularly experience increases in our net shipping cost due to co...</code>                               | <code>1.0</code> |
* Loss: [<code>BinaryCrossEntropyLoss</code>](https://sbert.net/docs/package_reference/cross_encoder/losses.html#binarycrossentropyloss) with these parameters:
  ```json
  {
      "activation_fn": "torch.nn.modules.linear.Identity",
      "pos_weight": null
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 3
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `project`: huggingface
- `trackio_space_id`: trackio
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: no
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: True
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Framework Versions
- Python: 3.11.9
- Sentence Transformers: 5.2.2
- Transformers: 4.57.1
- PyTorch: 2.2.2+cpu
- Accelerate: 1.12.0
- Datasets: 2.14.7
- Tokenizers: 0.22.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->