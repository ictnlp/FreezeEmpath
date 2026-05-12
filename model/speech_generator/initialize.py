import transformers
from transformers import AutoModel, Qwen2ForCausalLM, Qwen2Config, AutoTokenizer
from dataclasses import dataclass, field


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    base_model_type: str = field(
        default="llama",
        metadata={"help": "Type of the base model."}
    )
    unit_vocab_size: int = field(
        default=4096,
        metadata={"help": "Maximum unit value."}
    )
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."}
    )
    tokenizer_path: str = field(
        default=None,
        metadata={"help": "Path to the tokenizer."}
    )
    hidden_size: int = field(
        default=512,
        metadata={"help": "Hidden size of the model."}
    )
    num_hidden_layers: int = field(
        default=12,
        metadata={"help": "Number of hidden layers in the model."}
    )
    num_attention_heads: int = field(
        default=8,
        metadata={"help": "Number of attention heads in each layer."}
    )
    intermediate_size: int = field(
        default=2048,
        metadata={"help": "Intermediate size of the feed-forward layers."}
    )
    max_position_embeddings: int = field(
        default=4096,
        metadata={"help": "Maximum number of position embeddings."}
    )
    init_from_scratch: bool = field(
        default=False,
        metadata={"help": "Initialize the model from scratch."}
    )


def initialize_tokenizer(tokenizer_path, unit_vocab_size, max_position_embeddings):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        model_max_length=max_position_embeddings,
        padding_side="right",
        use_fast=False
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.add_tokens(["<sep>"])
    tokenizer.add_tokens([f"<{i}>" for i in range(unit_vocab_size)])

    return tokenizer


def initialize_model(tokenizer, model_name_or_path, hidden_size, num_hidden_layers, num_attention_heads, intermediate_size, max_position_embeddings, model_type="qwen", init_from_scratch=False):
    model_cls = Qwen2ForCausalLM
    cfg_cls = Qwen2Config
    if model_name_or_path:
        if init_from_scratch:
            config = cfg_cls.from_pretrained(model_name_or_path)
            model = AutoModel.from_config(config)
        else:
            model = model_cls.from_pretrained(model_name_or_path)
        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            model.resize_token_embeddings(len(tokenizer))
    else:
        config = cfg_cls(
            vocab_size=len(tokenizer),
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
        )
        model = model_cls(config)
    
    return model


def main():
    parser = transformers.HfArgumentParser((TrainingArguments,))
    args = parser.parse_args_into_dataclasses()[0]

    # Initialize tokenizer
    tokenizer = initialize_tokenizer(
        tokenizer_path=args.tokenizer_path,
        unit_vocab_size=args.unit_vocab_size,
        max_position_embeddings=args.max_position_embeddings
    )

    # Initialize model
    model = initialize_model(
        tokenizer=tokenizer,
        model_name_or_path=args.model_name_or_path,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.max_position_embeddings,
        model_type=args.base_model_type,
        init_from_scratch=args.init_from_scratch
    )

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
