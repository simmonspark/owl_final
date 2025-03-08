import torch
import torch.nn as nn
from transformers import OwlViTProcessor, OwlViTForObjectDetection, CLIPModel
import yaml
from src.dataset import get_dataloaders
from tqdm import tqdm
from PIL import Image

tmp = ['banner', 'bench', 'fence', 'flame', 'food_truck', 'garbage_bag',
       'park_headstone', 'park_info', 'park_pot', 'pet', 'rest_area',
       'sit_board', 'smoke', 'street_lamp', 'street_vendor', 'tent',
       'toilet', 'trash_can']

def get_training_config():
    with open("/media/sien/media/code/owl_final/config.yaml", "r") as stream:
        data = yaml.safe_load(stream)
        return data["training"]


def clip_train():
    from transformers import OwlViTConfig
    from transformers import AutoProcessor
    config_dict = OwlViTConfig.get_config_dict("google/owlvit-base-patch32")
    config = OwlViTConfig(config_dict[0])
    model = OwlViTModel(config)
    training_cfg = get_training_config()
    train_dataloader, test_dataloader, scales, labelmap = get_dataloaders()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=training_cfg["weight_decay"],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _processor = AutoProcessor.from_pretrained("google/owlvit-base-patch32")
    inputs = _processor(
        text=tmp,
        images=Image.new("RGB", (224, 224)),
        return_tensors="pt",
    )
    ids = inputs["input_ids"]
    model.train()
    model.to(device)
    patient = 3
    best = torch.inf
    while True:
        losses = []
        for i, (image, labels, boxes, metadata) in enumerate(
                tqdm(train_dataloader, ncols=60)
        ):
            store = 0
            text_token = ids.to(device)
            # logits_per_image : idx[0] : 1,3
            # logits_per_text : idx[1] : 3,1
            # text_embeds : idx[2] : 3,512
            # image_embeds : idx[3] : 1,512
            # text_model_output : idx[4] :
            # idx[0] : last_hidden_state -> shape(3,16,512)
            # idx[1] : pooled_output -> shape(3,512)
            # vision_model_output : idx[5] :
            # idx[0] : last_hidden_state : 1, 577, 768
            # idx[1] : pooler_output : 1,768
            optimizer.zero_grad()
            image = image.to(device)
            text_token = ids.to(device)
            out = model(input_ids=text_token, pixel_values=image)

            # img2text
            logits_img2text = out[0]
            probs_img2text = torch.softmax(logits_img2text, dim=-1)
            index = labels.squeeze(0)
            pos_probs_img = probs_img2text[0, index]
            neg_mask = torch.ones(probs_img2text.shape[-1], dtype=bool)
            neg_mask[index] = False
            neg_probs_img = probs_img2text[0, neg_mask]
            loss_img2text = -torch.log(pos_probs_img.sum() / (pos_probs_img.sum() + neg_probs_img.sum() + 1e-8))

            '''# text2img
            logits_text2img = out[1]  # (3, 1)
            probs_text2img = torch.softmax(logits_text2img, dim=-1)
            pos_probs_text = probs_text2img[index]  # (2, 1)
            neg_probs_text = probs_text2img[~neg_mask]  # (1, 1)
            loss_text2img = -torch.log(pos_probs_text.sum() / (pos_probs_text.sum() + neg_probs_text.sum() + 1e-8))

            # 총 손실
            loss = (loss_img2text + loss_text2img) / 2'''
            loss_img2text.backward()
            optimizer.step()
            store = store + loss_img2text.item()
        losses.append(store)
        _ = sum(losses) / len(losses)
        if _ < best:
            best = _
        else :
            patient -= 1
        if patient == 0:
            torch.save(model.state_dict(),'./clip_model.pth')




class OwlViTConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def set(self, key, value):
        setattr(self, key, value)


class Preprocessor:
    def __init__(self):
        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")

    def __call__(self, text, images, **kwargs):  # 키워드 통일
        '''
        input_ids : tokenized
        attention_mask : tokenized
        pixel_values 1, 3, 768, 768
        '''
        return self.processor(text=text, images=images, **kwargs)


class NewGELUActivation(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x * 1.702)


def contrastive_loss(logit):
    '''
    input : logit : B, T, T
    return : loss, tensor
    '''
    return nn.functional.cross_entropy(logit, torch.arange(len(logit)).to(logit.device))


def clip_loss(text2img, img2text):
    '''
    input : img & text : B,C : 이미지 또는 텍스트 쿼리를 대표하는 zip tensor
    return : loss
    batch
    '''
    text2img_loss = contrastive_loss(text2img)
    img2text_loss = contrastive_loss(img2text)
    return (text2img_loss + img2text_loss) / 2.0


class OwlViTMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = NewGELUActivation()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class OwlViTAttention(nn.Module):
    def __init__(self, config):
        super(OwlViTAttention, self).__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, input: torch.Tensor, T: int, B: int):
        return input.view(B, T, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states: torch.Tensor):
        bsz, tgt_len, embed_dim = hidden_states.size()
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_probs = attn_probs.to(value_states.dtype)
        attn_output = attn_probs @ value_states
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output


class OwlViTEncoderLayer(nn.Module):
    def __init__(self, config):
        super(OwlViTEncoderLayer, self).__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.self_attn = OwlViTAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = OwlViTMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor):
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class OwlViTEncoder(nn.Module):
    def __init__(self, config):
        '''
        just encoder encoder layers
        입력 shape이랑 출력 shape은 동일하게 유지
        '''
        super(OwlViTEncoder, self).__init__()
        self.config = config
        self.layers = nn.ModuleList([OwlViTEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, inputs_embeds):
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(hidden_states)
            hidden_states = layer_outputs
        return hidden_states


class OwlViTTextEmbeddings(nn.Module):
    def __init__(self, config):
        super(OwlViTTextEmbeddings, self).__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    def forward(self, input_ids=None) -> torch.Tensor:
        seq_length = input_ids.size(1)
        position_ids = self.position_ids[:, :seq_length]
        inputs_embeds = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings
        return embeddings


class OwlViTVisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.class_embedding = nn.Parameter(torch.randn(config.hidden_size))

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )

        self.num_patches = (config.image_size // config.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def forward(self, pixel_values) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [batch_size, num_channels, height, width]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)

        return embeddings


class OwlViTTextTransformer(nn.Module):
    def __init__(self, config):
        super(OwlViTTextTransformer, self).__init__()
        self.config = config.text_config
        embed_dim = config.hidden_size
        self.embeddings = OwlViTTextEmbeddings(config)
        self.encoder = OwlViTEncoder(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, input_ids):
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        hidden_states = self.embeddings(input_ids=input_ids)

        encoder_outputs = self.encoder(inputs_embeds=hidden_states)
        last_hidden_state = encoder_outputs
        last_hidden_state = self.final_layer_norm(last_hidden_state)
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            input_ids.to(torch.int).argmax(dim=-1).to(last_hidden_state.device),
        ]
        # contrastive learning에서 가장 마지막 token embedding을 씀
        # return
        # idx[0] : last_hidden_state -> shape(3,16,512)
        # idx[1] : pooled_output -> shape(3,512)
        return (last_hidden_state, pooled_output)


class OwlViTVisionTransformer(nn.Module):
    def __init__(self, config):
        super(OwlViTVisionTransformer, self).__init__()
        self.config = config
        self.embeddings = OwlViTVisionEmbeddings(config)
        self.pre_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.encoder = OwlViTEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.FloatTensor):
        expected_input_dtype = self.embeddings.patch_embedding.weight.dtype
        pixel_values = pixel_values.to(expected_input_dtype)

        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layernorm(hidden_states)

        encoder_outputs = self.encoder(inputs_embeds=hidden_states)

        last_hidden_state = encoder_outputs
        pooled_output = last_hidden_state[:, 0, :]

        pooled_output = self.post_layernorm(pooled_output)
        # contrastive learning에서 가장 첫번 째 vector를 씀
        # return
        # idx[0] : last_hidden_state : 1, 577, 768
        # idx[1] : pooler_output : 1,768
        return (last_hidden_state, pooled_output)


class OwlViTModel(nn.Module):
    def __init__(self, config):
        super(OwlViTModel, self).__init__()
        self.config = config
        text_config = config.text_config
        vision_config = config.vision_config
        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        self.text_model = OwlViTTextTransformer(text_config)
        self.vision_model = OwlViTVisionTransformer(vision_config)

        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(config.logit_scale_init_value))

    def forward(self, input_ids=None, pixel_values=None, loss=False):
        # idx[0] : last_hidden_state : 1, 577, 768
        # idx[1] : pooler_output : 1,768
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        # idx[0] : last_hidden_state -> shape(3,16,512)
        # idx[1] : pooled_output -> shape(3,512)
        text_outputs = self.text_model(input_ids=input_ids)

        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)
        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)
        image_embeds = image_embeds / torch.linalg.norm(image_embeds, ord=2, dim=-1, keepdim=True)
        text_embeds_norm = text_embeds / torch.linalg.norm(text_embeds, ord=2, dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp().to(image_embeds.device)

        logits_per_text = torch.matmul(text_embeds_norm, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.t()


        text_embeds = text_embeds_norm

        # return
        # logits_per_image : idx[0] : 1,3
        # logits_per_text : idx[1] : 3,1
        # text_embeds : idx[2] : 3,512
        # image_embeds : idx[3] : 1,512
        # text_model_output : idx[4] :
        # idx[0] : last_hidden_state -> shape(3,16,512)
        # idx[1] : pooled_output -> shape(3,512)
        # vision_model_output : idx[5] :
        # idx[0] : last_hidden_state : 1, 577, 768
        # idx[1] : pooler_output : 1,768
        # clip_loss : idx[6] : 그냥 contrastive에서만 쓰임
        return (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)


if __name__ == "__main__":
    clip_train()
