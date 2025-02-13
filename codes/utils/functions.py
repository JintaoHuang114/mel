import torch
import base64
from openai import OpenAI
from omegaconf import OmegaConf
from torch.nn.functional import softmax


def setup_parser():
    args = OmegaConf.load("../config/richpediamel.yaml")
    return args


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def help_indices_by_entropy(scores: torch.tensor, threshold: float):
    entropy = -torch.sum(softmax(scores, dim=1) * torch.log(softmax(scores, dim=1) + 1e-9), dim=1)
    indices = torch.where(entropy > threshold)
    return indices

def top1_by_LLM(mention_name, mention_text, mention_pic_path, candidate_dicts, mention_img_folder):
    if mention_pic_path != '' and mention_pic_path is not None:
        base64_image = encode_image(mention_img_folder + "/" + mention_pic_path)

    client = OpenAI(
        api_key="",
        base_url="https://open.bigmodel.cn/api/paas/v4/"
    )

    PROMPT = """
    ###Mention
    Name: {mention_name}
    Context: {mention_context}

    ###Entity table
    0. 
        Name:{name_0}
        Description:{desc_0}
    1. 
        Name:{name_1} 
        Description:{desc_1}
    2. 
        Name:{name_2}
        Description:{desc_2}
    3. 
        Name:{name_3}
        Description:{desc_3}
    4. 
        Name:{name_4}
        Description:{desc_4}
    5. 
        Name:{name_5}
        Description:{desc_5}
    6. 
        Name:{name_6}
        Description:{desc_6}
    7. 
        Name:{name_7}
        Description:{desc_7}
    8. 
        Name:{name_8}
        Description:{desc_8}
    9. 
        Name:{name_9}
        Description:{desc_9}

    Just give the serial number and do not give me any other information.
    The most matched serial number is:
    """
    PROMPT = PROMPT.format(mention_name=mention_name, mention_context=mention_text,
                           name_0=candidate_dicts[0]['name'], desc_0=candidate_dicts[0]['text'],
                           name_1=candidate_dicts[1]['name'], desc_1=candidate_dicts[1]['text'],
                           name_2=candidate_dicts[2]['name'], desc_2=candidate_dicts[2]['text'],
                           name_3=candidate_dicts[3]['name'], desc_3=candidate_dicts[3]['text'],
                           name_4=candidate_dicts[4]['name'], desc_4=candidate_dicts[4]['text'],
                           name_5=candidate_dicts[5]['name'], desc_5=candidate_dicts[5]['text'],
                           name_6=candidate_dicts[6]['name'], desc_6=candidate_dicts[6]['text'],
                           name_7=candidate_dicts[7]['name'], desc_7=candidate_dicts[7]['text'],
                           name_8=candidate_dicts[8]['name'], desc_8=candidate_dicts[8]['text'],
                           name_9=candidate_dicts[9]['name'], desc_9=candidate_dicts[9]['text'])
    if mention_pic_path != '' and mention_pic_path is not None:
        message = [
            {"role": "system",
             "content": "You are an expert in selecting the best-matched entity to match the given mention, note that the picture belongs to the mention. Compare names in mentions with each entity in the entity table and prioritize exact matches or the closest ones firstly, then check if the description of the entity is more closely related to the mention's context."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpg;base64,{base64_image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": PROMPT
                    },
                ],
            }
        ]
    else:
        message = [
            {"role": "system",
             "content": "You are an expert in selecting the best-matched entity to match the given mention. Compare names in mentions with each entity in the entity table and prioritize exact matches or the closest ones firstly, then check if the description of the entity is more closely related to the mention's context."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": PROMPT
                    },
                ],
            }
        ]
    response = client.chat.completions.create(
        model="glm-4v-flash",
        messages=message,
    )
    entity = response.choices[0].message.content
    return entity


def rerank_topk(rank: torch.Tensor, mention_index, top1index, topk_cddt_indices):
    if rank[mention_index, top1index].item() == 1:
        return rank
    else:
        current_rank = rank[mention_index, top1index].item()
        for _ in topk_cddt_indices:
            a = rank[mention_index, _].item()
            if a < current_rank:
                rank[mention_index, _] = a + 1
        rank[mention_index, top1index] = 1
        return rank