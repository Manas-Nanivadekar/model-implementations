import torch, torch.nn as nn
from transformers import CLIPModel, CLIPProcessor


hf_id = "openai/clip-vit-base-patch32"
clip = CLIPModel.from_pretrained(hf_id).eval().float()


class ClipImageEncoder(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    @torch.no_grad()
    def forward(self, pixel_values):
        out = self.m.vision_model(pixel_values=pixel_values)
        emb = self.m.visual_projection(out.pooler_output)
        return emb / emb.norm(dim=-1, keepdim=True)


class ClipTextEncoder(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    @torch.no_grad()
    def forward(self, input_ids, attention_mask):
        out = self.m.text_model(input_ids=input_ids, attention_mask=attention_mask)
        emb = self.m.text_projection(out.pooler_output)
        return emb / emb.norm(dim=-1, keepdim=True)


img_encoder = ClipImageEncoder(clip)
txt_encoder = ClipTextEncoder(clip)


ex_img = torch.randn(1, 3, 224, 224, dtype=torch.float32)
ex_ids = torch.ones(1, 77, dtype=torch.int64)
ex_mask = torch.ones(1, 77, dtype=torch.int64)


from torch import export as texport

exported_img = texport.export(img_encoder, (ex_img,))
exported_txt = texport.export(txt_encoder, (ex_ids, ex_mask))

from executorch.exir import to_edge_transform_and_lower

img_edge = to_edge_transform_and_lower(exported_img)
txt_edge = to_edge_transform_and_lower(exported_txt)

img_pte = img_edge.to_executorch()
txt_pte = txt_edge.to_executorch()

open("clip_image_encoder.pte", "wb").write(img_pte.buffer)
open("clip_text_encoder.pte", "wb").write(txt_pte.buffer)

print("Wrote clip_image_encoder.pte and clip_text_encoder.pte")
