from typing import List
import os
import numpy as np
import torch
import io
from PIL import Image, ImageDraw, ImageFont

# please make sure https://github.com/IDEA-Research/GroundingDINO is installed correctly.
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span



class VLMManager:
    def __init__(self, checkpoint_path='Open-GroundingDino/logs/checkpoint0001.pth', config_file = 'Open-GroundingDino/tools/GroundingDINO_SwinT_OGC.py'):
        
        self.model = self.load_model(config_file, checkpoint_path, cpu_only=False)

        pass

    def identify(self, image: bytes, caption: str) -> List[int]:
        image_pil = Image.open(io.BytesIO(image)).convert("RGB")
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        img, _ = transform(image_pil, None)  
        
        boxes_filt, pred_phrases = self.get_grounding_output(
            self.model, img, caption , 0.2, 0.2, cpu_only=False, token_spans=None
        )

        size = image_pil.size
        pred_dict = {
            "boxes": boxes_filt,
            "size": [size[1], size[0]],  # H,W
            "labels": pred_phrases,
        }
        if len(pred_dict['labels']) == 0:
            boxes_filt, pred_phrases = self.get_grounding_output(
            self.model, img, caption , 0.000001, 0.1, cpu_only=False, token_spans=None
        )
            size = image_pil.size
            pred_dict = {
            "boxes": boxes_filt,
            "size": [size[1], size[0]],  # H,W
            "labels": pred_phrases,
        }
        
        boxes = self.get_boxes(pred_dict)
        
        return boxes
    
    def get_boxes(self, tgt):
        H, W = tgt["size"]
        boxes = tgt["boxes"]
        labels = tgt["labels"]
        coll = []
        assert len(boxes) == len(labels), "boxes and labels must have same length"
        for box, label in zip(boxes, labels):
            # from 0..1 to 0..W, 0..H
            box = box * torch.Tensor([W, H, W, H])
            # from xywh to xyxy
            box[:2] -= box[2:] / 2
            box[2:] += box[:2]
            x0, y0, x1, y1 = box
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
            coll.append([x0, y0, x1-x0, y1-y0]) #
        if len(coll) == 0:
            return [0,0,0,0]
        else:
            return coll[np.argsort(labels)[-1]]


    

    def load_model(self,model_config_path, model_checkpoint_path, cpu_only=False):
        args = SLConfig.fromfile(model_config_path)
        args.device = "cuda" if not cpu_only else "cpu"
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)

        _ = model.eval()
        return model


    def get_grounding_output(self,model, image, caption, box_threshold, text_threshold=None, with_logits=True, cpu_only=False, token_spans=None):
        assert text_threshold is not None or token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
            
        device = "cuda" if not cpu_only else "cpu"
        model = model.to(device)
        image = image.to(device)
        with torch.no_grad():
            outputs = model(image[None], captions=[caption])
        logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"][0]  # (nq, 4)

        # filter output
        if token_spans is None:
            logits_filt = logits.cpu().clone()
            boxes_filt = boxes.cpu().clone()
            filt_mask = logits_filt.max(dim=1)[0] > box_threshold
            logits_filt = logits_filt[filt_mask]  # num_filt, 256
            boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

            # get phrase
            tokenlizer = model.tokenizer
            tokenized = tokenlizer(caption)
            # build pred
            pred_phrases = []
            for logit, box in zip(logits_filt, boxes_filt):
                pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
                if with_logits:
                    pred_phrases.append( logit.max().item() )
                else:
                    pred_phrases.append(pred_phrase)
        else:
            # given-phrase mode
            positive_maps = create_positive_map_from_span(
                model.tokenizer(text_prompt),
                token_span=token_spans
            ).to(image.device) # n_phrase, 256

            logits_for_phrases = positive_maps @ logits.T # n_phrase, nq
            all_logits = []
            all_phrases = []
            all_boxes = []
            for (token_span, logit_phr) in zip(token_spans, logits_for_phrases):
                # get phrase
                phrase = ' '.join([caption[_s:_e] for (_s, _e) in token_span])
                # get mask
                filt_mask = logit_phr > box_threshold
                # filt box
                all_boxes.append(boxes[filt_mask])
                # filt logits
                all_logits.append(logit_phr[filt_mask])
                if with_logits:
                    logit_phr_num = logit_phr[filt_mask]
                    all_phrases.extend([phrase + f"({str(logit.item())[:4]})" for logit in logit_phr_num])
                else:
                    all_phrases.extend([phrase for _ in range(len(filt_mask))])
            boxes_filt = torch.cat(all_boxes, dim=0).cpu()
            pred_phrases = all_phrases


        return boxes_filt, pred_phrases



# from typing import List
# import os
# import numpy as np
# import torch
# import io
# from PIL import Image, ImageDraw, ImageFont

# # please make sure https://github.com/IDEA-Research/GroundingDINO is installed correctly.
# import groundingdino.datasets.transforms as T
# from groundingdino.models import build_model
# from groundingdino.util import box_ops
# from groundingdino.util.slconfig import SLConfig
# from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
# from groundingdino.util.vl_utils import create_positive_map_from_span



# class VLMManager:
#     def __init__(self, checkpoint_path='Open-GroundingDino/logs/checkpoint0001.pth', config_file = 'Open-GroundingDino/tools/GroundingDINO_SwinT_OGC.py'):
        
#         self.model = self.load_model(config_file, checkpoint_path, cpu_only=False)

#         pass

#     def identify(self, image: bytes, caption: str) -> List[int]:
#         image_pil = Image.open(io.BytesIO(image)).convert("RGB")
#         transform = T.Compose(
#             [
#                 T.RandomResize([800], max_size=1333),
#                 T.ToTensor(),
#                 T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#             ]
#         )
#         img, _ = transform(image_pil, None)  
        
#         boxes_filt, pred_phrases = self.get_grounding_output(
#             self.model, img, caption , 0.2, 0.2, cpu_only=False, token_spans=None
#         )

#         size = image_pil.size
#         pred_dict = {
#             "boxes": boxes_filt,
#             "size": [size[1], size[0]],  # H,W
#             "labels": pred_phrases,
#         }
#         if len(pred_dict['labels']) == 0:
#             boxes_filt, pred_phrases = self.get_grounding_output(
#             self.model, img, caption , 0.01, 0.1, cpu_only=False, token_spans=None
#         )
#             size = image_pil.size
#             pred_dict = {
#             "boxes": boxes_filt,
#             "size": [size[1], size[0]],  # H,W
#             "labels": pred_phrases,
#         }
        
#         boxes = self.get_boxes(pred_dict)
        
#         return boxes
    
#     def get_boxes(self, tgt):
#         H, W = tgt["size"]
#         boxes = tgt["boxes"]
#         labels = tgt["labels"]
#         coll = []
#         assert len(boxes) == len(labels), "boxes and labels must have same length"
#         for box, label in zip(boxes, labels):
#             # from 0..1 to 0..W, 0..H
#             box = box * torch.Tensor([W, H, W, H])
#             # from xywh to xyxy
#             box[:2] -= box[2:] / 2
#             box[2:] += box[:2]
#             x0, y0, x1, y1 = box
#             x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
#             coll.append([x0, y0, x1-x0, y1-y0]) #
#         # if len(coll) == 0:
#         #     return [0,0,0,0]
#         # else:
#         return coll, labels
#     # [np.argsort(labels)[-1]]


    

#     def load_model(self,model_config_path, model_checkpoint_path, cpu_only=False):
#         args = SLConfig.fromfile(model_config_path)
#         args.device = "cuda" if not cpu_only else "cpu"
#         model = build_model(args)
#         checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
#         load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)

#         _ = model.eval()
#         return model


#     def get_grounding_output(self,model, image, caption, box_threshold, text_threshold=None, with_logits=True, cpu_only=False, token_spans=None):
#         assert text_threshold is not None or token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
#         caption = caption.lower()
#         caption = caption.strip()
#         if not caption.endswith("."):
#             caption = caption + "."
            
#         device = "cuda" if not cpu_only else "cpu"
#         model = model.to(device)
#         image = image.to(device)
#         with torch.no_grad():
#             outputs = model(image[None], captions=[caption])
#         logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
#         boxes = outputs["pred_boxes"][0]  # (nq, 4)

#         # filter output
#         if token_spans is None:
#             logits_filt = logits.cpu().clone()
#             boxes_filt = boxes.cpu().clone()
#             filt_mask = logits_filt.max(dim=1)[0] > box_threshold
#             logits_filt = logits_filt[filt_mask]  # num_filt, 256
#             boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

#             # get phrase
#             tokenlizer = model.tokenizer
#             tokenized = tokenlizer(caption)
#             # build pred
#             pred_phrases = []
#             for logit, box in zip(logits_filt, boxes_filt):
#                 pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
#                 if with_logits:
#                     pred_phrases.append(pred_phrase )
#                 else:
#                     pred_phrases.append(pred_phrase)
#         else:
#             # given-phrase mode
#             positive_maps = create_positive_map_from_span(
#                 model.tokenizer(text_prompt),
#                 token_span=token_spans
#             ).to(image.device) # n_phrase, 256

#             logits_for_phrases = positive_maps @ logits.T # n_phrase, nq
#             all_logits = []
#             all_phrases = []
#             all_boxes = []
#             for (token_span, logit_phr) in zip(token_spans, logits_for_phrases):
#                 # get phrase
#                 phrase = ' '.join([caption[_s:_e] for (_s, _e) in token_span])
#                 # get mask
#                 filt_mask = logit_phr > box_threshold
#                 # filt box
#                 all_boxes.append(boxes[filt_mask])
#                 # filt logits
#                 all_logits.append(logit_phr[filt_mask])
#                 if with_logits:
#                     logit_phr_num = logit_phr[filt_mask]
#                     all_phrases.extend([phrase + f"({str(logit.item())[:4]})" for logit in logit_phr_num])
#                 else:
#                     all_phrases.extend([phrase for _ in range(len(filt_mask))])
#             boxes_filt = torch.cat(all_boxes, dim=0).cpu()
#             pred_phrases = all_phrases


#         return boxes_filt, pred_phrases



