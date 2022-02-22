from transformers import DistilBertTokenizer, DistilBertModel
from transformers import ViTModel, ViTConfig
import torch.nn as nn

class CrossModalBERT(nn.Module):

    def __init__(self):
        super(CrossModalBERT, self).__init__()
        self.distil_bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.configuration = ViTConfig(image_size=145, num_channels=1)
        self.vit = ViTModel(configuration)

        self.linear1 = nn.Linear(1536, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, ids, mask, audio):
        bert_out = self.distil_bert(ids, mask)
        text_feat = bert_out.last_hidden_state[:, -1, :] # get bert last hidden state
        vit_out = self.vit(audio)
        audio_feat = vit_out.pooler_output
        cat_feat = torch.cat((audio_feat, text_feat), 1)
        x = self.linear1(cat_feat)
        sig = self.sigmoid(x)
        return sig
