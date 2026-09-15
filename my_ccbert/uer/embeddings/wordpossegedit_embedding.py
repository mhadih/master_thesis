import torch
import torch.nn as nn

from uer.embeddings.wordpos_embedding import WordPosEmbedding


class WordPosSegEditEmbedding(WordPosEmbedding):
    """
    BERT embedding consists of three parts:
    word embedding, position embedding, and segment embedding.
    """
    def __init__(self, args, vocab_size):
        super(WordPosSegEditEmbedding, self).__init__(args, vocab_size)
        self.segment_embedding = nn.Embedding(5, args.emb_size)    ## 0--> padding; 1 --> code changes;  2 --> text guidance; | 2 unused
        self.edit_type_embedding = nn.Embedding(10, args.emb_size) ## 10 kinds of edits: insert, delete, replace, equal | text-guide | special token | 4 unused
        # self.distinguish_before_after_layer = nn.Linear(2*args.emb_size, args.emb_size)

    def forward(self, src, seg):
        """
        Args:
            src_before: [batch_size x seq_length]
            src_after: [batch_size x seq_length]
            seg: [batch_size x seq_length]
            edit: [batch_size x seq_length]
        Returns:
            emb: [batch_size x seq_length x hidden_size]
        """
        src_before, src_after, edit_type = src
        word_before_emb = self.word_embedding(src_before)
        word_after_emb  = self.word_embedding(src_after)
        pos_emb = self.position_embedding(
            torch.arange(0, word_after_emb.size(1), device=word_after_emb.device, dtype=torch.long)
            .unsqueeze(0)
            .repeat(word_after_emb.size(0), 1)
        )
        seg_emb = self.segment_embedding(seg)
        edit_emb = self.edit_type_embedding(edit_type)
        ## FSE submission version
        emb = word_before_emb + word_after_emb  + pos_emb + seg_emb + edit_emb

        ## use MLP try to distinguish before&after
        # before_after_emb = torch.cat((word_before_emb, word_after_emb), -1)
        # before_after_emb = self.distinguish_before_after_layer(before_after_emb)
        # emb = before_after_emb + pos_emb + seg_emb + edit_emb

        if not self.remove_embedding_layernorm:
            emb = self.layer_norm(emb)
        emb = self.dropout(emb)
        return emb

