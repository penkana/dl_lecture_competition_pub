import re
import random
import time
import  pandas as pd
from PIL import Image
import numpy as np
import pandas
import math
import torch
import torch.nn as nn
from typing import Optional
from torch import Tensor
from torch.nn import functional as F
from torchvision import transforms
from transformers import  ViTMAEModel,ViTMAEForPreTraining
from transformers import ViltProcessor, ViltForQuestionAnswering
from transformers import ViTImageProcessor
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms.functional import to_tensor
from transformers import RobertaTokenizer, RobertaModel
from collections import Counter
from scipy.linalg import svd
#import os
#os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
ANSWER_SPACE = 40001

device = "cuda" if torch.cuda.is_available() else "cpu"

# BERTのトークナイザーを読み込み
tokenizer = RobertaTokenizer.from_pretrained('roberta-base',force_download=True)
# BERTモデルを読み込み
text_model = RobertaModel.from_pretrained('roberta-base',force_download=True).to(device)

#model_vit = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').to(device)

#model_mae = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-large").to(device)
feature_extractor = ViTImageProcessor.from_pretrained('facebook/vit-mae-base',force_download=True)

model_mae = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base",force_download=True).to(device)
#model_mae = ViTMAEModel.from_pretrained("facebook/vit-mae-large").to(device)

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa",force_download=True)
model_vilt = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa",force_download=True).to(device)
model_vilt.config.num_labels = ANSWER_SPACE
model_vilt.classifier = nn.Sequential(
    nn.Linear(in_features=768, out_features=6144, bias=True),
    nn.LayerNorm((6144,), eps=1e-05, elementwise_affine=True),
    nn.GELU(approximate='none'),
    nn.Linear(in_features=6144, out_features=ANSWER_SPACE, bias=True)
    ).to(device)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(1)

def process_text(text):
    # lowercase
    text = text.lower()

    # 数詞を数字に変換
    num_word_to_digit = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10'
    }
    for word, digit in num_word_to_digit.items():
        text = text.replace(word, digit)

    # 小数点のピリオドを削除
    text = re.sub(r'(?<!\d)\.(?!\d)', '', text)

    # 冠詞の削除
    text = re.sub(r'\b(a|an|the)\b', '', text)

    # 短縮形のカンマの追加
    contractions = {
        "dont": "don't", "isnt": "isn't", "arent": "aren't", "wont": "won't",
        "cant": "can't", "wouldnt": "wouldn't", "couldnt": "couldn't"
    }
    for contraction, correct in contractions.items():
        text = text.replace(contraction, correct)

    # 句読点をスペースに変換
    text = re.sub(r"[^\w\s':]", ' ', text)

    # 句読点をスペースに変換
    text = re.sub(r'\s+,', ',', text)

    # 連続するスペースを1つに変換
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# 1. データローダーの作成
class VQADataset(torch.utils.data.Dataset):
    def __init__(self, df_path, image_dir, transform=None, answer=True,vilt = False):
        self.transform = transform  # 画像の前処理
        self.image_dir = image_dir  # 画像ファイルのディレクトリ
        self.df = pandas.read_json(df_path)  # 画像ファイルのパス，question, answerを持つDataFrame
        self.answer = answer
        self.vilt = vilt

        # question / answerの辞書を作成
        self.question2idx = {}
        self.answer2idx = {}
        self.idx2question = {}
        self.idx2answer = {}

        # 質問文に含まれる単語を辞書に追加
        for question in self.df["question"]:
            question = process_text(question)
            words = tokenizer.tokenize(question)
            for word in words:
                if word not in self.question2idx:
                    self.question2idx[word] = tokenizer.convert_tokens_to_ids(word)
        self.idx2question = {v: k for k, v in self.question2idx.items()}  # 逆変換用の辞書(question)

        if self.answer:
            tmp_list = pd.Series([])
            for answers in self.df["answers"]:
                tmp = pd.Series([self.clean_answer(process_text(answer["answer"])) for answer in answers])
                tmp_list = pd.concat([tmp_list,tmp],ignore_index=True)
            self.total_answer_counts = tmp_list.value_counts()
            #self.total_answer_counts = self.df["answers"].explode().value_counts()
            print("The total count of answer types are: {}".format(self.total_answer_counts.shape))
            print("Picking the top {} as answer space (and +1 for unknown).".format(ANSWER_SPACE-1))

            self.answer_space = self.total_answer_counts.head(ANSWER_SPACE-1).index.to_list()
            self.answer_space.append("<Unknown>")

        """
        if self.answer:
            # 回答に含まれる単語を辞書に追加
            for answers in self.df["answers"]:
                for answer in answers:
                    word = answer["answer"]
                    if word == "unsuitable image":
                        word = "unsuitable"
                    word = process_text(word)
                    if word not in self.answer2idx:
                        self.answer2idx[word] = len(self.answer2idx)
            for word in self.df_ans["answer"]:
                word = process_text(word)
                if word not in self.answer2idx:
                    self.answer2idx[word] = len(self.answer2idx)
            self.answer2idx["<Unknown>"] = len(self.answer2idx)
            self.idx2answer = {v: k for k, v in self.answer2idx.items()}  # 逆変換用の辞書(answer)
        """

        #https://www.kaggle.com/code/lhanhsin/vizwiz-vqa-example
        if self.answer:
            # 回答に含まれる単語を辞書に追加
            for word in self.answer_space:
                if word not in self.answer2idx:
                    self.answer2idx[word] = len(self.answer2idx)
            self.idx2answer = {v: k for k, v in self.answer2idx.items()}  # 逆変換用の辞書(answer)

            self.freq_weight = 1/np.log(self.total_answer_counts.head(ANSWER_SPACE-1).values+10)
            self.freq_weight = torch.tensor(np.append(self.freq_weight/np.mean(self.freq_weight),[0]),dtype=torch.float)

    def choose_ans_label(self,ans_list):
        ans_dict = Counter(ans_list)
        max_entry = "<Unknown>"
        max_count = 0
        for k,v in ans_dict.items():
            if k in self.answer_space and v>max_count:
                max_count = v
                max_entry = k
        return max_entry

    def choose_ans(self,ans_list):
        len_ = len(ans_list)
        tmp = [k for k in ans_list if k in self.answer_space]
        tmp += ["<Unknown>"]*(len_-len(tmp))
        return tmp

    def clean_answer(self,word):
        if word == "unsuitable image":
            return "unsuitable"
        else:
            return word

    def update_dict(self, dataset):
        """
        検証用データ，テストデータの辞書を訓練データの辞書に更新する．

        Parameters
        ----------
        dataset : Dataset
            訓練データのDataset
        """
        self.question2idx = dataset.question2idx
        self.answer2idx = dataset.answer2idx
        self.idx2question = dataset.idx2question
        self.idx2answer = dataset.idx2answer

    def __getitem__(self, idx):
        """
        対応するidxのデータ（画像，質問，回答）を取得．

        Parameters
        ----------
        idx : int
            取得するデータのインデックス

        Returns
        -------
        image : torch.Tensor  (C, H, W)
            画像データ
        question : torch.Tensor  (vocab_size)
            質問文をone-hot表現に変換したもの
        answers : torch.Tensor  (n_answer)
            10人の回答者の回答のid
        mode_answer_idx : torch.Tensor  (1)
            10人の回答者の回答の中で最頻値の回答のid
        """
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}")
        image = self.transform(image)
        #image = apply_zca_whitening(image, self.ZCA_matrix, self.mean)
        text = process_text(self.df["question"][idx])

        if not self.vilt:
            inputs = tokenizer(text, return_tensors="pt", max_length=17, truncation=True, padding="max_length").to(device)
            text_model.eval()
            encoded_questions = []
            with torch.no_grad():
                outputs = text_model(**inputs)
                #question = outputs.pooler_output.squeeze()
                question = outputs.last_hidden_state.squeeze()
                encoded_questions.append(question)
            question = torch.stack(encoded_questions).squeeze()
        else:
            question = text

        if self.vilt:
            if self.answer:
                answers = [process_text(answer["answer"]) for answer in self.df["answers"][idx]]
                mode_answer_idx = self.answer2idx[self.choose_ans_label(answers)]
                answers_ = self.choose_ans(answers)
                answers_ = [self.answer2idx[word] for word in answers_]
                target = torch.zeros(ANSWER_SPACE)
                for i in answers_:
                    target[i] += 1
                target = (target / sum(target)).to(device)
                return image, question, torch.Tensor(target),torch.tensor(answers_) ,int(mode_answer_idx)

            else:
                return image, question
        else:
            if self.answer:
                answers = [process_text(answer["answer"]) for answer in self.df["answers"][idx]]
                mode_answer_idx = self.answer2idx[self.choose_ans_label(answers)]
                answers_ = self.choose_ans(answers)
                answers_ = [self.answer2idx[word] for word in answers_]
                target = torch.zeros(ANSWER_SPACE)
                for i in answers_:
                    target[i] += 1
                target = (target / sum(target)).to(device)

                return image, torch.Tensor(question), torch.Tensor(target),torch.Tensor(answers_), int(mode_answer_idx)

            else:
                return image, torch.Tensor(question)

    def __len__(self):
        return len(self.df)

# 2. 評価指標の実装
# 簡単にするならBCEを利用する
def VQA_criterion(batch_pred: torch.Tensor, batch_answers: torch.Tensor):
    total_acc = 0.

    for pred, answers in zip(batch_pred, batch_answers):
        acc = 0.
        for i in range(len(answers)):
            num_match = 0
            for j in range(len(answers)):
                if i == j:
                    continue
                if pred == answers[j]:
                    num_match += 1
            acc += min(num_match / 3, 1)
        total_acc += acc / 10

    return total_acc / len(batch_pred)

# 3. モデルのの実装
# ResNetを利用できるようにしておく
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += self.shortcut(residual)
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, layers[0], 64)
        self.layer2 = self._make_layer(block, layers[1], 128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], 256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 768)
        self.bn2 = nn.BatchNorm1d(768)

    def _make_layer(self, block, blocks, out_channels, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        x = self.bn2(x)

        return x

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def ResNet50():
    return ResNet(BottleneckBlock, [3, 8,36, 3])

class BILSTM(nn.Module):
    def __init__(self,emb_dim, hid_dim):
        super().__init__()
        self.emb = nn.Embedding(emb_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, 2,dropout = 0.4,batch_first=True,bidirectional=True)  # nn.LSTMの使用
        self.linear = nn.Linear(hid_dim,768)

        self.norm0 = nn.LayerNorm(emb_dim)
        self.norm5 = nn.LayerNorm(768)

        self.hidden_size = hid_dim

        torch.nn.init.kaiming_normal_(self.linear.weight)
        torch.nn.init.normal_(self.linear.bias,std=0.03)

    def forward(self, x):
        h = self.norm0(x)
        h , (b,c) = self.lstm(h)
        y = self.linear(b[-1,:,:])
        y = self.norm5(y)
        return y

def extract_features(model, inputs):
    model.eval()
    with torch.no_grad():
        outputs = model(inputs,output_hidden_states=True)
        outputs = outputs.hidden_states[-1]
        #print(outputs.shape)
    return outputs

#https://yuiga.dev/blog/posts/vilbert/
class CoAttention(nn.Module):
    def __init__(self, text_embed_dim=768,image_embed_dim=768,dropout=0.3, dim_feedforward = 1024,num_heads=8):
        super(CoAttention, self).__init__()
        self.text_multihead_attn = nn.MultiheadAttention(text_embed_dim, num_heads,dropout=0.2,batch_first=True)
        self.image_multihead_attn = nn.MultiheadAttention(image_embed_dim, num_heads,dropout=0.2,batch_first=True)
        self.text_norm = nn.LayerNorm(text_embed_dim)
        self.image_norm = nn.LayerNorm(image_embed_dim)
        self.text_dropout = nn.Dropout(dropout)
        self.image_dropout = nn.Dropout(dropout)
        self.text_linear1 = nn.Linear(text_embed_dim, dim_feedforward)
        self.text_dropout2 = nn.Dropout(dropout)
        self.text_linear2 = nn.Linear(dim_feedforward, text_embed_dim)
        self.image_linear1 = nn.Linear(image_embed_dim, dim_feedforward)
        self.image_dropout2 = nn.Dropout(dropout)
        self.image_linear2 = nn.Linear(dim_feedforward, image_embed_dim)
        self.text_dropout3 = nn.Dropout(dropout)
        self.image_dropout3 = nn.Dropout(dropout)
        self.text_norm2 = nn.LayerNorm(text_embed_dim)
        self.image_norm2 = nn.LayerNorm(image_embed_dim)

        torch.nn.init.kaiming_normal_(self.text_linear1.weight)
        torch.nn.init.normal_(self.text_linear1.bias,std=0.03)
        torch.nn.init.kaiming_normal_(self.text_linear2.weight)
        torch.nn.init.normal_(self.text_linear2.bias,std=0.03)
        torch.nn.init.kaiming_normal_(self.image_linear1.weight)
        torch.nn.init.normal_(self.image_linear1.bias,std=0.03)
        torch.nn.init.kaiming_normal_(self.image_linear2.weight)
        torch.nn.init.normal_(self.image_linear2.bias,std=0.03)

    def forward(self, image_feature, text_feature):
        text_attn_output, _ = self.text_multihead_attn(text_feature, image_feature, image_feature)
        image_attn_output, _ = self.image_multihead_attn(image_feature, text_feature, text_feature)

        text_feature = text_feature+self.text_dropout(text_attn_output)
        image_feature = image_feature+self.image_dropout(image_attn_output)

        text_feature = self.text_norm(text_feature)
        image_feature = self.image_norm(image_feature)

        text_feature2 = self.text_linear2(self.text_dropout2(F.relu(self.text_linear1(text_feature))))
        image_feature2 = self.image_linear2(self.image_dropout2(F.relu(self.image_linear1(image_feature))))

        text_feature = text_feature + self.text_dropout3(text_feature2)
        image_feature = image_feature + self.image_dropout3(image_feature2)

        text_feature = self.text_norm2(text_feature)
        image_feature = self.image_norm2(image_feature)
        return image_feature,text_feature

"""
class ImageAttention(nn.Module):
    def __init__(self):
        super(ImageAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=768,num_heads=8,dropout=0.2,batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.5)
        self.norm = nn.LayerNorm(768)
        self.norm2 = nn.LayerNorm(768)
        self.lstm = BILSTM(768,768)
        self.linear1 = nn.Linear(768,768)
        self.linear2 = nn.Linear(768,768)
        self.fc = nn.Linear(768,ANSWER_SPACE)

    def forward(self,image,text):
        x,_ = self.attention(image,text,text)
        image = image + self.dropout(x)
        image = self.norm(image)

        image2 = self.linear2(self.dropout2(F.relu(self.linear1(image))))
        image = image + self.dropout3(image2)
        image = self.norm2(image)

        image = self.lstm(image)
        image = self.fc(image)
        return image
"""

class VQAModel(nn.Module):
    def __init__(self, n_answer,encoder,res):
        super().__init__()
        self.resnet = ResNet50()
        self.lstm1 = BILSTM(768,768)
        self.lstm2 = BILSTM(768,768)
        self.nn1 = nn.Linear(768, 768)
        self.nn2 = nn.Linear(768+768, 768+768)
        self.nn3 = nn.Linear(768, 768)
        self.image_encoder = encoder
        self.attention = CoAttention(text_embed_dim=768,image_embed_dim=768,dropout=0.5).to(device)
        self.bool_ = res
        #self.img_attention = ImageAttention()

        self.fc = nn.Sequential(
            nn.Linear(768*4, n_answer),
        )
        for m in self.fc:
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                torch.nn.init.normal_(m.bias,std=0.03)

        torch.nn.init.kaiming_normal_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias,std=0.03)
        torch.nn.init.kaiming_normal_(self.nn2.weight)
        torch.nn.init.normal_(self.nn2.bias,std=0.03)
        torch.nn.init.kaiming_normal_(self.nn3.weight)
        torch.nn.init.normal_(self.nn3.bias,std=0.03)

    def forward(self, image, question):
        #51415
        question_feature = question
        if self.bool_:
            image_feature = self.resnet(image)  # 画像の特徴量
        else:
            image_feature = extract_features(self.image_encoder,image).to(device)
            image_feature_atten,question_feature_atten = self.attention(image_feature,question_feature)
            image_feature *= image_feature_atten
            question_feature *= question_feature_atten
            image_feature = self.lstm2(image_feature)
            #x = self.img_attention(image_feature,question_feature)

        question_feature = self.lstm1(question_feature) # テキストの特徴量
        mix1 = torch.cat((image_feature, question_feature), dim=1)
        x1 = self.nn1(question_feature)
        x2 = self.nn2(mix1)
        x3 = self.nn3(image_feature)
        mix = torch.cat((x1,x2,x3),dim=1)
        x = self.fc(mix)

        return x

def mask_image(image, mask_ratio=0.75):
    # 画像の一部をランダムにマスク
    mask = np.random.rand(*image.shape) < mask_ratio
    masked_image = image * ~mask
    return masked_image, mask

# 4. 学習の実装
def pre_train(dataloader,net):
    #criterion = torch.nn.MSELoss()
    net.train()
    num_epochs = 5
    lr = 0.001
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr*0.001)
    # トレーニングループ
    for epoch in range(num_epochs):
        start = time.time()
        running_loss = 0.0
        for inputs, question, target,answers, mode_answer in dataloader:
            # 画像の一部をマスク
            masked_inputs, masks = [], []
            for pre_image in inputs:
                masked_image, mask = mask_image(pre_image.to('cpu').detach().numpy().copy())
                masked_inputs.append(torch.tensor(masked_image))
                masks.append(torch.tensor(mask))

            img = torch.stack(masked_inputs)
            img = img.float().to(device)
            outputs = net(img).logits.to(device)
            inputs = inputs.to(device)
            masks = torch.stack(masks)
            outputs = net.unpatchify(outputs)

            loss = torch.mean((outputs.to(device) - inputs) ** 2 * masks.to(device)) / 0.75
            #loss = criterion(outputs,inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(dataloader.dataset)
        if epoch % 1 == 0 or epoch == 0:
            print(f'Epoch {epoch+1}/{num_epochs} Loss: {epoch_loss:.4f} Time: {(time.time()-start):.2f} [s]')
        scheduler.step()

def pre_train_vilt(dataloader,net,criterion,initial_seed):
    net.train()
    num_epochs = 3
    lr = 0.00005
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr,weight_decay=0.005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr*0.001)
    # トレーニングループ
    for epoch in range(num_epochs):
        set_seed(initial_seed+epoch)
        start = time.time()
        running_loss = 0.0
        total_acc = 0

        for inputs, target,answers, mode_answer in dataloader:
            optimizer.zero_grad()
            inputs = inputs.to(device)
            #mode_answer = F.one_hot(mode_answer, num_classes=ANSWER_SPACE).float().to(device)
            outputs = net(**inputs).logits.to(device)
            outputs = F.log_softmax(outputs,dim=1)
            loss = criterion(outputs, target)
            #loss = criterion(outputs, mode_answer)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_acc += VQA_criterion(outputs.argmax(1), answers)  # VQA accuracy
        epoch_loss = running_loss / len(dataloader)
        total_acc = total_acc / len(dataloader)
        if epoch % 1 == 0 or epoch == 0:
            print(f'Epoch {epoch+1}/{num_epochs} Loss: {epoch_loss:.4f} train acc: {total_acc:.4f} Time: {(time.time()-start):.2f} [s]')
        scheduler.step()

def train(model, dataloader, optimizer, scheduler,criterion, device):
    model.train()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for image, question, target, answers,mode_answer in dataloader:
        optimizer.zero_grad()
        image, question, answer, mode_answer = \
            image.to(device), question.to(device), target.to(device), mode_answer.to(device)

        pred = model(image, question).to(device)
        pred = F.log_softmax(pred,dim=1)
        #mode_answer = F.one_hot(mode_answer, num_classes=ANSWER_SPACE).float().to(device)
        #loss = criterion(pred, mode_answer)
        loss = criterion(pred,target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        #simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()  # simple accuracy
        simple_acc = 0.0
    scheduler.step()
    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start

def eval(model, dataloader, optimizer,scheduler, criterion, device):
    model.eval()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for image, question, answers, mode_answer in dataloader:
        image, question, answer, mode_answer = \
            image.to(device), question.to(device), answers.to(device), mode_answer.to(device)

        pred = model(image, question)
        loss = criterion(pred, mode_answer.squeeze())

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()  # simple accuracy

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start

def GridMaskF( img):
        #if np.random.rand() > 0.5:return img

        h, w = img.size
        mask = np.ones((h, w), np.float32)

        for _ in range(4):
            d = np.random.randint(min(h, w) // 4)
            d = max(1, d)
            l = np.random.randint(0, w - d)
            r = np.random.randint(l + d, w)
            t = np.random.randint(0, h - d)
            b = np.random.randint(t + d, h)
            mask[t:b, l:r] = 0

        img = to_tensor(img)
        mask = torch.from_numpy(mask)
        img *= mask.unsqueeze(0)
        img = to_pil_image(img)
        return img

def main():
    set_seed(42)
    image_size = 224
    size = (image_size,image_size)
    # dataloader / model
    transform = transforms.Compose([
        transforms.Resize(size=size,antialias=True),
        transforms.ToTensor()
    ])
    dataset = VQADataset(df_path="../data/train.json", image_dir="../data/train", transform=transform)
    test_dataset = VQADataset(df_path="../data/valid.json", image_dir="../data/valid", transform=transform, answer=False)
    test_dataset.update_dict(dataset)

    pre_transform_center = transforms.Compose([transforms.ToPILImage(),transforms.RandomHorizontalFlip(p=0.5),transforms.CenterCrop(size=(30, 30))])
    pre_transform_normal = transforms.Compose([transforms.ToPILImage()])

    transform_GridMask = transforms.Compose([transforms.ToPILImage(),transforms.RandomHorizontalFlip(p=0.5),GridMaskF])
    transform_normal = transforms.Compose([transforms.ToPILImage()])
    transform_Adjust = transforms.Compose([transforms.ToPILImage(),transforms.RandomHorizontalFlip(p=0.5),transforms.RandomAdjustSharpness(sharpness_factor=5, p=0.5)])

    pre_transform_train = transforms.RandomChoice([pre_transform_normal,pre_transform_center])

    transform_train = transforms.RandomChoice([transform_Adjust,transform_normal,transform_GridMask])
    transform = transforms.Compose([transforms.ToPILImage()])

    def worker_init_fn(worker_id):
        set_seed(np.random.get_state()[1][0] + worker_id)

    # ミニバッチごとにリアルタイムで変換するcollate_fnの定義
    def real_time_transform_collate_fn(batch):
        torch.cuda.empty_cache()
        images, question,target,answers,mode_answer_idx= list(zip(*batch))  # リスト内の各要素をタプルとして扱う
        images = torch.stack(images)
        question = torch.stack(question)
        target = torch.stack(target)
        answers = torch.stack(answers)
        mode_answer_idx = torch.tensor(mode_answer_idx)
        # 画像を変換
        transformed_images = [transform_train(image) for image in images]
        transformed_images = feature_extractor(images=transformed_images, return_tensors="pt")
        transformed_images = transformed_images["pixel_values"]
        return transformed_images, question,target,answers,mode_answer_idx

    def real_time_transform_collate_fn_pre(batch):
        torch.cuda.empty_cache()
        images, question,target,answers,mode_answer_idx= list(zip(*batch))  # リスト内の各要素をタプルとして扱う
        images = torch.stack(images)
        # 画像を変換
        transformed_images = [pre_transform_train(image) for image in images]
        transformed_images = feature_extractor(images=transformed_images, return_tensors="pt")
        transformed_images = transformed_images["pixel_values"]
        return transformed_images, question,target,answers,mode_answer_idx

    def real_time_transform_collate_fn_test(batch):
        torch.cuda.empty_cache()
        images, question= list(zip(*batch))  # リスト内の各要素をタプルとして扱う
        images = torch.stack(images)
        question = torch.stack(question)
        # 画像を変換
        transformed_images = [transform(image) for image in images]
        transformed_images = feature_extractor(images=transformed_images, return_tensors="pt")
        transformed_images = transformed_images["pixel_values"]
        return transformed_images, question

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True,worker_init_fn = worker_init_fn,collate_fn=real_time_transform_collate_fn,drop_last=True)
    pre_train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True,collate_fn=real_time_transform_collate_fn_pre,drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32*2, shuffle=False,collate_fn=real_time_transform_collate_fn_test)

    print("pre train...")

    #pre_train(pre_train_loader,model_mae)
    #model_mae.save_pretrained("./finetuned_vit")

    model_mae = ViTMAEForPreTraining.from_pretrained("./finetuned_vit")

    model = VQAModel(n_answer=len(dataset.answer2idx),encoder=model_mae,res=False).to(device)

    num_epoch = 3
    lr = 0.0001
    #51354
    criterion = nn.KLDivLoss(reduction="batchmean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=lr*0.001)
    print("train start...")
    # train model
    for epoch in range(num_epoch):
        set_seed(epoch)
        train_loss, train_acc, train_simple_acc, train_time = train(model, train_loader, optimizer,scheduler, criterion, device)
        print(f"【{epoch + 1}/{num_epoch}】\n"
                f"train time: {train_time:.2f} [s]\n"
                f"train loss: {train_loss:.4f}\n"
                f"train acc: {train_acc:.4f}"
                #f"train simple acc: {train_simple_acc:.4f}"
                )
    print("train fin...")
    torch.save(model.state_dict(), "model.pth")
    # 提出用ファイルの作成
    model.eval()
    submission = []
    start = time.time()
    with torch.no_grad():
        for image, question in test_loader:
            image, question = image.to(device), question.to(device)
            predictions = []
            predictions.append(model(image,question))
            predictions_tensor = torch.stack(predictions)
            final_pred = torch.mean(input=predictions_tensor, dim=0)
            for i in range(len(final_pred)):
                tmp = final_pred[i].argmax().item()
                if dataset.idx2answer[tmp] == "<Unknown>":
                    final_pred[i][tmp] = -9999
            final_pred = final_pred.argmax(1).tolist()
            submission.extend(final_pred)
    print("eval fin...")
    print(f"eval time:{(time.time()-start):.4f}")
    submission = [dataset.idx2answer[id] for id in submission]
    submission = np.array(submission)
    np.save("submission.npy", submission)
"""
def main3():
    set_seed(42)
    image_size = 224
    size = (image_size,image_size)
    # dataloader / model
    transform = transforms.Compose([
        transforms.Resize(size=size,antialias=True),
        transforms.ToTensor()
    ])
    dataset = VQADataset(df_path="../data/train.json", image_dir="../data/train", transform=transform,vilt = True)
    test_dataset = VQADataset(df_path="../data/valid.json", image_dir="../data/valid", transform=transform, answer=False,vilt = True)
    test_dataset.update_dict(dataset)

    pre_transform_rotate = transforms.Compose([transforms.ToPILImage(),transforms.RandomHorizontalFlip(p=0.5),transforms.RandomRotation(degrees=(-20, 20))])
    #pre_transform_center = transforms.Compose([transforms.ToPILImage(),transforms.RandomHorizontalFlip(p=0.5),transforms.CenterCrop(size=(30, 30))])
    #pre_transform_affine = transforms.Compose([transforms.ToPILImage(),transforms.RandomHorizontalFlip(p=0.5),transforms.RandomAffine(degrees=0, translate=(0.3, 0.5))])
    #pre_transform_Perspective = transforms.Compose([transforms.ToPILImage(),transforms.RandomHorizontalFlip(p=0.5),transforms.RandomPerspective(distortion_scale=0.5, p=1.0)])
    #pre_transform_GridMask = transforms.Compose([transforms.ToPILImage(),transforms.RandomHorizontalFlip(p=0.5),GridMaskF])
    pre_transform_normal = transforms.Compose([transforms.ToPILImage()])
    #pre_transform_gray =  transforms.Compose([transforms.ToPILImage(),transforms.RandomHorizontalFlip(p=0.5),transforms.RandomGrayscale(p=1)])
    #pre_transform_Adjust = transforms.Compose([transforms.ToPILImage(),transforms.RandomHorizontalFlip(p=0.5),transforms.RandomAdjustSharpness(sharpness_factor=5, p=0.5)])
    pre_transform_affine_rotate = transforms.Compose([transforms.ToPILImage(),transforms.RandomHorizontalFlip(p=0.5),transforms.RandomRotation(degrees=(-20, 20)),transforms.RandomAffine(degrees=0, translate=(0.3, 0.5))])

    transform_train = transforms.RandomChoice([pre_transform_normal,pre_transform_rotate])
    transform = transforms.Compose([transforms.ToPILImage()])

    def worker_init_fn(worker_id):
        set_seed(np.random.get_state()[1][0] + worker_id)

    # ミニバッチごとにリアルタイムで変換するcollate_fnの定義
    def real_time_transform_collate_fn(batch):
        torch.cuda.empty_cache()
        images, question,target,answers,mode_answer_idx= list(zip(*batch))  # リスト内の各要素をタプルとして扱う
        images = torch.stack(images)
        #question = torch.stack(question)
        target = torch.stack(target)
        answers = torch.stack(answers)
        mode_answer_idx = torch.tensor(mode_answer_idx)
        # 画像を変換
        transformed_images = [transform_train(image) for image in images]
        inputs = processor(transformed_images, question, return_tensors="pt",max_length=17, padding=True, truncation=True)
        return inputs,target,answers,mode_answer_idx

    def real_time_transform_collate_fn_test(batch):
        torch.cuda.empty_cache()
        images, question= list(zip(*batch))  # リスト内の各要素をタプルとして扱う
        images = torch.stack(images)
        #question = torch.stack(question)
        # 画像を変換
        transformed_images = [transform(image) for image in images]
        inputs = processor(transformed_images, question, return_tensors="pt",max_length=17, padding=True, truncation=True)
        return inputs

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True,worker_init_fn = worker_init_fn,collate_fn=real_time_transform_collate_fn,drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32*2, shuffle=False,collate_fn=real_time_transform_collate_fn_test)

    print("train...")
    #criterion = FocalLoss(alpha=dataset.freq_weight,gamma=2,reduction='mean',label_smoothing=0.2)
    criterion = nn.KLDivLoss(reduction="batchmean")
    pre_train_vilt(train_loader,model_vilt,criterion)
    model_vilt.save_pretrained("./finetuned_vilt")

    #model_vilt = ViltForQuestionAnswering.from_pretrained("./finetuned_vilt")
    print("train fin...")
    # 提出用ファイルの作成
    model_vilt.eval()
    submission = []
    start = time.time()
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            predictions = []
            outputs = model_vilt(**inputs).logits.to(device)
            predictions.append(outputs)
            predictions_tensor = torch.stack(predictions)
            final_pred = torch.mean(input=predictions_tensor, dim=0)
            for i in range(len(final_pred)):
                tmp = final_pred[i].argmax().item()
                if dataset.idx2answer[tmp] == "<Unknown>":
                    final_pred[i][tmp] = -9999
            final_pred = final_pred.argmax(1).tolist()
            submission.extend(final_pred)
    print("eval fin...")
    print(f"eval time:{(time.time()-start):.4f}")
    submission = [dataset.idx2answer[id] for id in submission]
    submission = np.array(submission)
    np.save("submission.npy", submission)
"""
def main_multi():
    # 3 603
    initial_seed = 4
    set_seed(initial_seed)
    image_size = 224
    size = (image_size,image_size)
    # dataloader / model
    transform = transforms.Compose([
        transforms.Resize(size=size,antialias=True),
        transforms.ToTensor()
    ])
    dataset = VQADataset(df_path="../data/train.json", image_dir="../data/train", transform=transform)
    test_dataset = VQADataset(df_path="../data/valid.json", image_dir="../data/valid", transform=transform, answer=False)
    test_dataset.update_dict(dataset)
    vilt_dataset = VQADataset(df_path="../data/train.json", image_dir="../data/train", transform=transform,vilt = True)
    vilt_test_dataset = VQADataset(df_path="../data/valid.json", image_dir="../data/valid", transform=transform, answer=False,vilt = True)
    vilt_test_dataset.update_dict(vilt_dataset)

    pre_transform_center = transforms.Compose([transforms.ToPILImage(),transforms.RandomHorizontalFlip(p=0.5),transforms.CenterCrop(size=(30, 30))])
    pre_transform_normal = transforms.Compose([transforms.ToPILImage()])

    transform_GridMask = transforms.Compose([transforms.ToPILImage(),transforms.RandomHorizontalFlip(p=0.5),GridMaskF])
    transform_normal = transforms.Compose([transforms.ToPILImage()])
    transform_Adjust = transforms.Compose([transforms.ToPILImage(),transforms.RandomHorizontalFlip(p=0.5),transforms.RandomAdjustSharpness(sharpness_factor=5, p=0.5)])

    vilt_pre_transform_rotate = transforms.Compose([transforms.ToPILImage(),transforms.RandomHorizontalFlip(p=0.5),transforms.RandomRotation(degrees=(-20, 20))])
    vilt_pre_transform_normal = transforms.Compose([transforms.ToPILImage()])

    pre_transform_train = transforms.RandomChoice([pre_transform_normal,pre_transform_center])

    transform_train = transforms.RandomChoice([transform_Adjust,transform_normal,transform_GridMask])
    transform = transforms.Compose([transforms.ToPILImage()])
    vilt_transform_train = transforms.RandomChoice([vilt_pre_transform_normal,vilt_pre_transform_rotate])


    def worker_init_fn(worker_id):
        set_seed(np.random.get_state()[1][0] + worker_id)

    # ミニバッチごとにリアルタイムで変換するcollate_fnの定義
    def real_time_transform_collate_fn(batch):
        torch.cuda.empty_cache()
        images, question,target,answers,mode_answer_idx= list(zip(*batch))  # リスト内の各要素をタプルとして扱う
        images = torch.stack(images)
        question = torch.stack(question)
        target = torch.stack(target)
        answers = torch.stack(answers)
        mode_answer_idx = torch.tensor(mode_answer_idx)
        # 画像を変換
        transformed_images = [transform_train(image) for image in images]
        transformed_images = feature_extractor(images=transformed_images, return_tensors="pt")
        transformed_images = transformed_images["pixel_values"]
        return transformed_images, question,target,answers,mode_answer_idx

    def real_time_transform_collate_fn_pre(batch):
        torch.cuda.empty_cache()
        images, question,target,answers,mode_answer_idx= list(zip(*batch))  # リスト内の各要素をタプルとして扱う
        images = torch.stack(images)
        # 画像を変換
        transformed_images = [pre_transform_train(image) for image in images]
        transformed_images = feature_extractor(images=transformed_images, return_tensors="pt")
        transformed_images = transformed_images["pixel_values"]
        return transformed_images, question,target,answers,mode_answer_idx

    def real_time_transform_collate_fn_test(batch):
        torch.cuda.empty_cache()
        images, question= list(zip(*batch))  # リスト内の各要素をタプルとして扱う
        images = torch.stack(images)
        question = torch.stack(question)
        # 画像を変換
        transformed_images = [transform(image) for image in images]
        transformed_images = feature_extractor(images=transformed_images, return_tensors="pt")
        transformed_images = transformed_images["pixel_values"]
        return transformed_images, question

    def vilt_real_time_transform_collate_fn(batch):
        torch.cuda.empty_cache()
        images, question,target,answers,mode_answer_idx= list(zip(*batch))  # リスト内の各要素をタプルとして扱う
        images = torch.stack(images)
        #question = torch.stack(question)
        target = torch.stack(target)
        answers = torch.stack(answers)
        mode_answer_idx = torch.tensor(mode_answer_idx)
        # 画像を変換
        transformed_images = [vilt_transform_train(image) for image in images]
        inputs = processor(transformed_images, question, return_tensors="pt",max_length=17, padding=True, truncation=True)
        return inputs,target,answers,mode_answer_idx

    def vilt_real_time_transform_collate_fn_test(batch):
        torch.cuda.empty_cache()
        images, question= list(zip(*batch))  # リスト内の各要素をタプルとして扱う
        images = torch.stack(images)
        #question = torch.stack(question)
        # 画像を変換
        transformed_images = [transform(image) for image in images]
        inputs = processor(transformed_images, question, return_tensors="pt",max_length=17, padding=True, truncation=True)
        return inputs

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True,worker_init_fn = worker_init_fn,collate_fn=real_time_transform_collate_fn,drop_last=True)
    pre_train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True,collate_fn=real_time_transform_collate_fn_pre,drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32*2, shuffle=False,collate_fn=real_time_transform_collate_fn_test)

    vilt_train_loader = torch.utils.data.DataLoader(vilt_dataset, batch_size=16, shuffle=True,worker_init_fn = worker_init_fn,collate_fn=vilt_real_time_transform_collate_fn,drop_last=True)
    vilt_test_loader = torch.utils.data.DataLoader(vilt_test_dataset, batch_size=32*2, shuffle=False,collate_fn=vilt_real_time_transform_collate_fn_test)

    print("pre train...")
    #pre_train(pre_train_loader,model_mae)
    #model_mae.save_pretrained("./finetuned_vit")
    model_mae = ViTMAEForPreTraining.from_pretrained("./finetuned_vit")
    print("pre train fin...")

    print("train start...")
    num_epoch = 3
    #num_epoch2 = 10
    lr = 0.0001
    #lr2 = 0.001
    model = VQAModel(n_answer=len(dataset.answer2idx),encoder=model_mae,res=False).to(device)
    #model2 = VQAModel(n_answer=len(dataset.answer2idx),encoder=model_mae,res=True).to(device)
    criterion = nn.KLDivLoss(reduction="batchmean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=lr*0.001)
    #optimizer2 = torch.optim.AdamW(model2.parameters(), lr=lr2, weight_decay=0.005)
    #scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=num_epoch, eta_min=lr2*0.001)
    # train model
    """
    for epoch in range(num_epoch2):
        set_seed(epoch)
        train_loss, train_acc, train_simple_acc, train_time = train(model2, train_loader, optimizer2,scheduler2, criterion, device)
        print(f"【{epoch + 1}/{num_epoch2}】\n"
                f"train time: {train_time:.2f} [s]\n"
                f"train loss: {train_loss:.4f}\n"
                f"train acc: {train_acc:.4f}"
                #f"train simple acc: {train_simple_acc:.4f}"
                )
    """
    for epoch in range(num_epoch):
        set_seed(initial_seed+epoch)
        train_loss, train_acc, train_simple_acc, train_time = train(model, train_loader, optimizer,scheduler, criterion, device)
        print(f"【{epoch + 1}/{num_epoch}】\n"
                f"train time: {train_time:.2f} [s]\n"
                f"train loss: {train_loss:.4f}\n"
                f"train acc: {train_acc:.4f}"
                #f"train simple acc: {train_simple_acc:.4f}"
                )
    #torch.save(model.state_dict(), "model.pth")
    #model.load_state_dict(torch.load("model.pth"))
    print("train fin...")
    print("pre vilt train...")

    #0.5897
    vilt_criterion = nn.KLDivLoss(reduction="batchmean")
    pre_train_vilt(vilt_train_loader,model_vilt,vilt_criterion,initial_seed)
    #model_vilt.save_pretrained("./finetuned_vilt")

    print("pre vilt train fin...")


    # 提出用ファイルの作成
    model.eval()
    #model2.eval()
    model_vilt.eval()
    submission = []
    vilt_submission = []
    #start = time.time()
    with torch.no_grad():
        for image, question in test_loader:
            image, question = image.to(device), question.to(device)
            predictions = []
            #ans = (model(image,question) + model2(image,question)) / 2
            ans = model(image,question)
            predictions.append(ans)
            predictions_tensor = torch.stack(predictions)
            final_pred = torch.mean(input=predictions_tensor, dim=0)
            for i in range(len(final_pred)):
                tmp = final_pred[i].argmax().item()
                if dataset.idx2answer[tmp] == "<Unknown>":
                    final_pred[i][tmp] = -9999
            #final_pred = final_pred.argmax(1).tolist()
            submission.extend(final_pred)
    with torch.no_grad():
        for inputs in vilt_test_loader:
            inputs = inputs.to(device)
            predictions = []
            outputs = model_vilt(**inputs).logits.to(device)
            predictions.append(outputs)
            predictions_tensor = torch.stack(predictions)
            final_pred = torch.mean(input=predictions_tensor, dim=0)
            for i in range(len(final_pred)):
                tmp = final_pred[i].argmax().item()
                if dataset.idx2answer[tmp] == "<Unknown>":
                    final_pred[i][tmp] = -9999
            #final_pred = final_pred.argmax(1).tolist()
            vilt_submission.extend(final_pred)
    print("eval fin...")
    #print(f"eval time:{(time.time()-start):.4f}")
    #print(vilt_submission)
    #print(submission)
    submission = torch.stack(submission)
    vilt_submission = torch.stack(vilt_submission)
    final_submission = submission.to('cpu').detach().numpy().copy()*0.1 + 0.9 *vilt_submission.to('cpu').detach().numpy().copy()
    final_submission = final_submission.argmax(1).tolist()
    final_submission = [dataset.idx2answer[id] for id in final_submission]
    final_submission = np.array(final_submission)
    np.save("submission.npy", final_submission)
if __name__ == "__main__":
    #main()
    #main3()
    main_multi()
