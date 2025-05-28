from PIL import Image
import torch,open_clip
import numpy as np
import os
device='cuda' if torch.cuda.is_available() else 'cpu'
def cos_similary(a,b):
    return (a@b)/(np.linalg.norm(a)*np.linalg.norm(b))
def load_model():
    model_path=r"C:\Users\B237\Desktop\models"
    model_name='ViT-L-14'
    model,_,preprocess=open_clip.create_model_and_transforms(model_name)
    tokenizer=open_clip.get_tokenizer(model_name)
    ckpt=torch.load(f"{model_path}/RemoteCLIP-{model_name}.pt",map_location=torch.device('cpu'))
    model.load_state_dict(ckpt)
    model.to(device=device)
    return model,tokenizer,preprocess
def compute(model,tokenizer,preprocess,image_files,input_text):
    text_token=tokenizer([f'a satellite photo of {i}' for i in input_text]).to(device=device)
    text2vec=model.encode_text(text_token)
    yp=[]
    for i in os.listdir(image_files):
        image=Image.open(os.path.join(image_files,i))
        image=preprocess(image).unsqueeze(0).to(device=device)
        image_encode=model.encode_image(image)
        middle=[]
        for text_encode in text2vec:
            like=cos_similary(image_encode.cpu().detach().numpy(),
                                        text_encode.cpu().detach().numpy())
            middle.append(like)
        yp.append(input_text[np.argmax(middle)])
    return yp
def main():
    model,tokenizer,preprocess=load_model()
    yp=compute(model=model,tokenizer=tokenizer,preprocess=preprocess,image_files=r"C:\Users\B237\Desktop\分类",input_text=['beach','mountain','playground'])