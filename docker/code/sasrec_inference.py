# Copyright (c) 2024, ZDF.
from recommenders.models.sasrec.model import SASREC
import tensorflow as tf

class SARSEC_inference(SASREC):
    """
    Infernece SARSEC model for predictions of all the items for which 
    model has been trained
    """
    
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)

    def predict(self, inputs):
         
        training = False
        input_seq = inputs["input_seq"]
        candidate = inputs["candidate"]

        mask = tf.expand_dims(tf.cast(tf.not_equal(input_seq, 0), tf.float32), -1)
        seq_embeddings, positional_embeddings = self.embedding(input_seq)
        seq_embeddings += positional_embeddings
        # seq_embeddings = self.dropout_layer(seq_embeddings)
        seq_embeddings *= mask
        seq_attention = seq_embeddings
        seq_attention = self.encoder(seq_attention, training, mask)
        seq_attention = self.layer_normalization(seq_attention)  # (b, s, d)
        seq_emb = tf.reshape(
        seq_attention,
        [tf.shape(input_seq)[0] * self.seq_max_len, self.embedding_dim],
        )  # (b*s, d)
        candidate_emb = self.item_embedding_layer(candidate)  # (b, s, d)
        candidate_emb = tf.transpose(candidate_emb, perm=[0, 2, 1])  # (b, d, s)

        test_logits = tf.matmul(seq_emb, candidate_emb)
        # (200, 100) * (1, 101, 100)'

        test_logits = tf.reshape(
        test_logits,
        [tf.shape(input_seq)[0], self.seq_max_len, self.item_num],
        )  # (1, 200, 101)
        test_logits = test_logits[:, -1, :]  # (1, item_num)
        return test_logits
