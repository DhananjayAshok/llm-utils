import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, BitsAndBytesConfig, AutoModelForSequenceClassification, AutoModelForImageTextToText, AutoProcessor
from peft import LoraConfig, TaskType, get_peft_model
from training.data import infer_label_list



def get_model_processor(script_args, dataset):
    """
    Load the model and processor (tokenizer for LMs) with bits and bytes set up for the given script arguments. 

    If classification then handles the label list and config for the model.
    """
    bnb_config = None
    if script_args.use_bnb:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True, # set storage type
            bnb_4bit_quant_storage=torch.bfloat16, # set storage type
        )
    model_kwargs = {}
    if not script_args.distributed: # then we distribute the model ourselves
        model_kwargs["device_map"] = "auto"
    if script_args.training_kind != "clf":
        config = AutoConfig.from_pretrained(script_args.model_name, trust_remote_code=True)
        if script_args.modality == "lm":
            base_model = AutoModelForCausalLM.from_pretrained(
                script_args.model_name,
                config=config,
                quantization_config=bnb_config,
                trust_remote_code=True,
                **model_kwargs
            )
        elif script_args.modality == "vlm":
            base_model = AutoModelForImageTextToText.from_pretrained(
                script_args.model_name,
                config=config,
                quantization_config=bnb_config,
                trust_remote_code=True,
                **model_kwargs
            )
        else:
            raise ValueError(f"BROSKI WHAT IS THIS MODALITY: {script_args.modality}. Only lm and vlm are supported.")
    else:
        label_list = infer_label_list(dataset, script_args.parameters)
        num_labels = len(label_list)
        config = AutoConfig.from_pretrained(
            script_args.model_name,
            num_labels=num_labels,
            trust_remote_code=True,
            finetuning_task="text-classification")
        base_model = AutoModelForSequenceClassification.from_pretrained(
            script_args.model_name,
            config=config,
            trust_remote_code=True,
            quantization_config=bnb_config,
            **model_kwargs
        )
        label_to_id = {v: i for i, v in enumerate(label_list)}
        # update config with label infos
        base_model.config.label2id = label_to_id
        base_model.config.id2label = {id: label for label, id in label_to_id.items()}
    base_model.config.use_cache = False

    if script_args.modality == "lm":
        processor = AutoTokenizer.from_pretrained(script_args.model_name, trust_remote_code=True)
        tokenizer = processor
    elif script_args.modality == "vlm":
        processor = AutoProcessor.from_pretrained(script_args.model_name, trust_remote_code=True)
        tokenizer = processor.tokenizer
        processor.pad_token = processor.tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        base_model.config.pad_token_id = tokenizer.eos_token_id
        #tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
    return base_model, processor


def get_peft_config(script_args):
    """
    Get the PEFT config for the given script arguments.

    TODO: Customize the target modules based on the model architecture. Currently only proj modules are targeted and only LlaMa is supported.
    """
    task = TaskType.CAUSAL_LM
    if script_args.training_kind == "clf":
        task = TaskType.SEQ_CLS
    target_modules = "all-linear"
    if script_args.lora_target_modules is not None:
        target_modules = script_args.lora_target_modules
    peft_config = LoraConfig(
    r=script_args.lora_r,
    lora_alpha=script_args.lora_alpha,
    lora_dropout=script_args.lora_dropout,
    target_modules=target_modules,
    bias="none",
    task_type=task
    )
    return peft_config


def get_peft_model_processor(script_args, dataset):
    """
    Load the model and tokenizer with PEFT set up for the given script arguments.
    """
    base_model, processor = get_model_processor(script_args, dataset)
    peft_config = get_peft_config(script_args)
    model = get_peft_model(base_model, peft_config)
    return model, processor


def log_trainable_parameters(model, logger):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logger.info(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )