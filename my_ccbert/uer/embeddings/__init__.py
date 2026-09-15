from uer.embeddings.dual_embedding import DualEmbedding
from uer.embeddings.word_embedding import WordEmbedding
from uer.embeddings.wordpos_embedding import WordPosEmbedding
from uer.embeddings.wordposseg_embedding import WordPosSegEmbedding
from uer.embeddings.wordsinusoidalpos_embedding import WordSinusoidalposEmbedding
from uer.embeddings.wordpossegedit_embedding import WordPosSegEditEmbedding


str2embedding = {"word": WordEmbedding, "word_pos": WordPosEmbedding, "word_pos_seg": WordPosSegEmbedding,
                 "word_sinusoidalpos": WordSinusoidalposEmbedding, "dual": DualEmbedding, "word_pos_seg_edit": WordPosSegEditEmbedding}

__all__ = ["WordEmbedding", "WordPosEmbedding", "WordPosSegEmbedding", "WordSinusoidalposEmbedding",
           "DualEmbedding", "WordPosSegEditEmbedding",  "str2embedding"]

