# Copyright (c) 2024, ZDF.
from recommenders.models.sasrec.model import SASREC
from recommenders.utils.timer import Timer

import sys
import tensorflow as tf
import logging
import numpy as np
import sys

from tqdm import tqdm

class CustomSASRec(SASREC):
    """
    The inherited class of the SASREC Model, same as SASREC

    SAS Rec model
    Self-Attentive Sequential Recommendation Using Transformer

    :Citation:

        Wang-Cheng Kang, Julian McAuley (2018), Self-Attentive Sequential
        Recommendation. Proceedings of IEEE International Conference on
        Data Mining (ICDM'18)

        Original source code from nnkkmto/SASRec-tf2,
        https://github.com/nnkkmto/SASRec-tf2

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.epochs_loss_tracker = [] #To track the loss
    
    def loss_function_bce(self, pos_logits, neg_logits, istarget, epsilon_value):
        """Losses are calculated separately for the positive and negative
        items based on the corresponding logits. A mask is included to
        take care of the zero items (added for padding).

        Args:
            pos_logits (tf.Tensor): Logits of the positive examples.
            neg_logits (tf.Tensor): Logits of the negative examples.
            istarget (tf.Tensor): Mask for nonzero targets.
            epsilon_value (float): Value inlcuded to compute the loss function

        Returns:
            float: Loss.
        """

        pos_logits = pos_logits[:, 0]
        neg_logits = neg_logits[:, 0]

        # for logits
        loss = tf.reduce_sum(
            -tf.math.log(tf.math.sigmoid(pos_logits) + epsilon_value) * istarget
            - tf.math.log(1 - tf.math.sigmoid(neg_logits) + epsilon_value) * istarget
        ) / tf.reduce_sum(istarget)

        reg_loss = tf.compat.v1.losses.get_regularization_loss()
        
        loss += reg_loss

        return loss
    
    def loss_function_gbce(self, pos_logits, neg_logits, istarget, epsilon_value, neg_items_batch, calibration_gbce, total_items):
        """
        Computes the gbce loss function

        Args:
        pos_logits (tf.tensor): Logits of the positive examples.
        neg_logits (tf.Tensor): Logits of the negative examples.
        istarget (tf.Tensor): Mask for nonzero targets.
        epsilon_value (float): Value inlcuded to compute the loss function.
        neg_items_batch (tf.tensor) : Batch of negative_items.
        calibration_gbce (float): Constant value employed in gbce loss function, inorder to control the power parameter beta .
        total_items (int): Total item num in the dataset.

        Returns:
            float: Loss.
        
        """

        #Calculate the sampling rate(alpha) (number of negative items in the batch /total number of items in the dataset)
        alpha  = tf.math.count_nonzero(neg_items_batch, dtype=tf. float32)  / tf.constant(total_items, dtype=tf.float32)

        t_value = tf.constant(calibration_gbce, dtype=tf.float32)

        #Calculate power parameter(beta) using the equation.
        beta = alpha * (t_value* (1 - (1 / alpha)) + (1 / alpha))

        #Apply the power parameter(beta) on the bce loss function to control the scores of positive items (gbce loss function)
        pos_logits = pos_logits[:, 0]
        neg_logits = neg_logits[:, 0]

        # Apply loss value with power paraemters Beta
        loss = tf.reduce_sum(
            -tf.math.log(tf.math.pow(tf.math.sigmoid(pos_logits) + epsilon_value, beta)) * istarget
            - tf.math.log(1 - tf.math.sigmoid(neg_logits) + epsilon_value) * istarget
        ) / tf.reduce_sum(istarget)

        reg_loss = tf.compat.v1.losses.get_regularization_loss()
        
        loss += reg_loss

        return loss

    def top_k_function_tron(self, neg_logits, topk_value=100):

        # Get the indices of the top k values
        top_indices = tf.math.top_k(tf.squeeze(neg_logits), k=topk_value).indices

        # Create a mask where only the top-k values are set to 1
        mask = tf.zeros_like(tf.squeeze(neg_logits), dtype=tf.float32)
        mask = tf.tensor_scatter_nd_update(mask, tf.expand_dims(top_indices, axis=1), tf.ones_like(top_indices, dtype=tf.float32))

        # Expand dimensions to match the input tensor shape
        mask = tf.expand_dims(mask, axis=1)

        # Apply the mask to the input tensor
        masked_input = neg_logits * mask

        return masked_input
    
    def train(self, dataset, sampler, hyperparams):

        if hyperparams.loss_function == "gbce":
            logging.info(f"Using '{hyperparams.loss_function}' loss function for model training")
        
        else: 
            logging.info(f"Using the 'bce' loss function for model training as the loss function is set to '{hyperparams.loss_function}' ")

        #For logging the details of application of top_k_function
        if hyperparams.apply_topk_function:
            logging.info(f"Applying top_k function, with the top_k value set to {hyperparams.top_k_negatives}")
        
        else:
            logging.info(f"Ignoring top_k function as 'apply_topk_function' set to {hyperparams.apply_topk_function}")

        epsilon_loss_value_tf = tf.constant(hyperparams.epsilon_loss_value, dtype=tf.float32)

        num_steps = int(len(dataset.user_data) / hyperparams.batch_size)

        # If the num_steps are zero, abort the training and check the usage data
        if num_steps == 0:
            raise ValueError(f"The num_steps are {num_steps}, please check if the usage data is larger than the batch size. Aborting the training")

        if  hyperparams.gradient_clipping:
            optimizer = tf.keras.optimizers.Adam(learning_rate=hyperparams.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-7, clipnorm=1.0)

        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=hyperparams.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-7)

        train_loss = tf.keras.metrics.Mean(name="train_loss")

        train_step_signature = [
            {
                "users": tf.TensorSpec(shape=(None, 1), dtype=tf.int64),
                "input_seq": tf.TensorSpec(
                    shape=(None, self.seq_max_len), dtype=tf.int64
                ),
                "positive": tf.TensorSpec(
                    shape=(None, self.seq_max_len), dtype=tf.int64
                ),
                "negative": tf.TensorSpec(
                    shape=(None, self.seq_max_len), dtype=tf.int64
                ),
            },
            tf.TensorSpec(shape=(None, 1), dtype=tf.int64),
            tf.TensorSpec(shape=(), dtype=tf.float32)
        ]

        @tf.function(input_signature=train_step_signature)
        def train_step(inp, tar, epsilon_loss_value_tf):
            with tf.GradientTape() as tape:
                pos_logits, neg_logits, loss_mask = self(inp, training=True)

                 #Use appropriate loss function
                if hyperparams.loss_function == "gbce":
                    loss = self.loss_function_gbce(pos_logits, neg_logits, loss_mask, epsilon_loss_value_tf, inp["negative"], hyperparams.calibration_gbce, dataset.itemnum)
                    
                else:
                    loss = self.loss_function_bce(pos_logits, neg_logits, loss_mask, epsilon_loss_value_tf)
                #Apply Top K function 
                if hyperparams.apply_topk_function:
                    try:
                        neg_logits = self.top_k_function_tron(neg_logits, hyperparams.top_k_negatives)   

                    except Exception as e:
                            logging.error(f"Top_k value is set to {hyperparams.top_k_negatives}. Try setting the top_k value to lower value, Exception: {e}")
                            sys.exit(1)

            gradients = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.trainable_variables))

            train_loss(loss)
            return loss, gradients, pos_logits, neg_logits, loss_mask

        for epoch in range(1,  hyperparams.num_epochs + 1):
            
            # Calualte_loss list for logging the loss for each and every epoch.
            step_loss = []
            train_loss.reset_states()
            for step in tqdm(
                range(num_steps), total=num_steps, ncols=70, leave=False, unit="b"
            ):

                u, seq, pos, neg = sampler.next_batch()

                inputs, target = self.create_combined_dataset(u, seq, pos, neg)

                loss, gradients, pos_logits, neg_logits, loss_mask = train_step(inputs, target, epsilon_loss_value_tf)
                step_loss.append(loss)

                #For every batch within the epoch, check for the models inputs and ouptuts for the NaN values
                if tf.math.is_nan(loss):
                    logging.info(f"Epoch {epoch}, Step {step}, Loss is NaN")

                   #Inputs
                    if np.isnan(inputs['input_seq']).any():
                        logging.info("The user sequences has NaN values")
                    
                    if np.isnan(inputs['positive']).any() :
                        logging.info("The pos sequnces has NaN values")
                    
                    if np.isnan(inputs['negative']).any():
                        logging.info("The neg sequnces has NaN values")
                    
                    #Outputs
                    #Check for all outputs if there exists NaN values
                    has_nan_pos_logits = tf.math.is_nan(pos_logits)
                    has_nan_neg_logits = tf.math.is_nan(neg_logits)
                    has_nan_mask_values = tf.math.is_nan(loss_mask)

                    #Compute for all elements within the tensor and return  single boolean value True when naN exists
                    if tf.math.reduce_any(has_nan_pos_logits):
                        logging.info("The positve logits have NaN values")

                    if tf.math.reduce_any(has_nan_neg_logits):
                        logging.info("The Negative logits have NaN values")

                    if tf.math.reduce_any(has_nan_mask_values):
                        logging.info("The elements for the loss masked has NaN values")
                    
                    raise ValueError(f"For the epoch number {epoch} and {step} the loss is NaN. Aborting the training.")

            #Calculate the mean loss value for all the batches
            epoch_loss = tf.reduce_mean(step_loss, axis=0, keepdims=False)

            self.epochs_loss_tracker.append(epoch_loss)
            logging.info(f"For the epoch number {epoch}, loss during the training is {epoch_loss}")
