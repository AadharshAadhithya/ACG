from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch import nn
import einops
import contextlib
from Qformer import BertConfig, BertLMHeadModel
from typing import List
from config import config


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class BaseModel(nn.Module):
    def __init__(self, max_frame_pos=128, num_features=1024, device="cuda"):
        super().__init__()
        self.device = device

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.language_model.tokenizer_name)
        self.tokenizer.add_tokens(["[PLAYER]", "[TEAM]", "([TEAM])"], special_tokens=True)

        # LLM initialization
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            config.model.language_model.llm_name, torch_dtype=torch.bfloat16
        )
        self.llm_model.resize_token_embeddings(len(self.tokenizer))
        self.eos_token_id = self.tokenizer("\n", add_special_tokens=False).input_ids[0]

        # Positional embedding for video frames
        self.video_frame_position_embedding = nn.Embedding(max_frame_pos, num_features)

        # Vision LayerNorm
        self.ln_vision = LayerNorm(num_features)

        # Move to device
        self._move_to_device_p()

    def _move_to_device_p(self):
        self.llm_model = self.llm_model.to(self.device)
        for name, param in self.llm_model.named_parameters():
            param.requires_grad = False

        self.video_frame_position_embedding = self.video_frame_position_embedding.to(self.device)
        self.ln_vision = self.ln_vision.to(self.device).eval()

    @classmethod
    def init_video_Qformer(cls, num_query_token, vision_width, num_hidden_layers=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = vision_width
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token

        qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(torch.zeros(1, num_query_token, encoder_config.hidden_size))
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return qformer, query_tokens

    def get_input_embeddings(self, input_ids):
        # Override this in subclasses for custom embedding logic
        return self.llm_model.get_input_embeddings()(input_ids)

    def maybe_autocast(self, dtype=torch.float16):
        enable_autocast = self.device != torch.device("cpu")
        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()


class MyModel(BaseModel):
    def __init__(self, max_frame_pos=128, window=30, num_query_tokens=32, num_video_query_token=32, num_features=1024, device="cuda", inference=False):
        super().__init__(max_frame_pos=max_frame_pos, num_features=num_features, device=device)

        self.window = window
        self.num_query_tokens = num_query_tokens
        self.num_video_query_token = num_video_query_token
        self.inference = inference

        # Initialize video Q-Former before calling _move_to_device
        self.video_Qformer, self.video_query_tokens = self.init_video_Qformer(
            num_query_token=num_video_query_token, vision_width=num_features, num_hidden_layers=2
        )
        
        # Ensure Q-Former is properly configured
        self._configure_video_Qformer()

        # LLM projection
        self.llm_proj = nn.Linear(self.video_Qformer.config.hidden_size, 512)

        # Move all components to device
        self._move_to_device()

    def _configure_video_Qformer(self):
        self.video_Qformer.cls = None
        self.video_Qformer.bert.embeddings.word_embeddings = None
        self.video_Qformer.bert.embeddings.position_embeddings = None
        for layer in self.video_Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

    def _move_to_device(self):
        super()._move_to_device_p()
        self.video_Qformer = self.video_Qformer.to(self.device)
        self.video_query_tokens = self.video_query_tokens.to(self.device)
        self.llm_proj = self.llm_proj.to(self.device)

    def forward(self, batch):
        # Video features
        video_features = batch["vid_features"].to(self.device)
        batch_size, time_length, _, _ = video_features.size()
        video_features = self.ln_vision(video_features)
        video_features = einops.rearrange(video_features, "b t n f -> (b t) n f", b=batch_size, t=time_length)

        # Positional embeddings
        position_ids = torch.arange(time_length, dtype=torch.long, device=self.device).unsqueeze(0).expand(batch_size, -1)
        frame_position_embeddings = self.video_frame_position_embedding(position_ids).unsqueeze(-2)
        frame_hidden_state = einops.rearrange(video_features, "(b t) n f -> b t n f", b=batch_size, t=time_length)
        frame_hidden_state = frame_position_embeddings + frame_hidden_state
        frame_hidden_state = einops.rearrange(frame_hidden_state, "b t q h -> b (t q) h", b=batch_size, t=time_length)
        frame_atts = torch.ones(frame_hidden_state.size()[:-1], dtype=torch.long).to(self.device)

        # Q-Former processing
        video_query_tokens = self.video_query_tokens.expand(frame_hidden_state.shape[0], -1, -1)
        video_query_output = self.video_Qformer.bert(
            query_embeds=video_query_tokens,
            encoder_hidden_states=frame_hidden_state,
            encoder_attention_mask=frame_atts,
            return_dict=True,
        )
        video_hidden = video_query_output.last_hidden_state
        inputs_llm = self.llm_proj(video_hidden)

        # Language model inputs
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        inputs_embeds = self.get_input_embeddings(input_ids)

        # LLM forward
        with self.maybe_autocast():
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )
        return {"loss": outputs.loss}


from dataset import ACGDataset
from config import config


# Load dataset
ds = ACGDataset(config.data.train_path)

# Sample a batch from the dataset
batch = [ds[0], ds[1], ds[3], ds[4]]

# Collate the batch
collated_batch = ds.collator(batch)

# Initialize the model
model = MyModel(
    max_frame_pos=128,
    window=30,
    num_query_tokens=32,
    num_video_query_token=32,
    num_features=1024,
    device="cuda",  # Change to "cpu" if no GPU is available
    inference=False
)

# Perform a forward pass
out = model(collated_batch)

# Print the output
print(out)
