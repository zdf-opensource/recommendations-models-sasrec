# Copyright (c) 2024, ZDF.
# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.


import json
import os
import pickle
import flask
import numpy as np
import logging
from itertools import islice
from operator import itemgetter

from pa_base.models.base_model import normalize_scores

from sasrec_inference import SARSEC_inference

MODEL_DIR = "/opt/ml/model/"


class ModelService:
    """A singleton for holding the model. This simply loads the model and holds it.
    It has a predict function that does a prediction based on the model and the input data."""

    @staticmethod
    def get_model_and_meta():
        """Get the model object for this instance, loading it's weights if it's not already loaded."""
        #model = keras.models.load_model(os.path.join(MODEL_DIR, 'model_sasrec/'))
        weights_path = os.path.join(MODEL_DIR,'sasrec_weights/')

        # Load .pkl file for obtaining the number of items with model has been trained with and also include the mx_seq_length
        meta_file = os.path.join(MODEL_DIR, "meta.pkl")
        with open(meta_file, "rb") as file:
            meta = pickle.load(file)
        
        item_id_for_extid = meta["item_id_for_extid"]

        item_for_cf_id = meta["external_id_for_item_id"]
  
        model = SARSEC_inference(item_num =meta["num_items_training"], seq_max_len = meta["Sasrec_model_descritption"]["maxlen"],
         embedding_dim = meta["Sasrec_model_descritption"]["hidden_units"], attention_dim = meta["Sasrec_model_descritption"]["hidden_units"],
         conv_dims = [ meta["Sasrec_model_descritption"]["hidden_units"],  meta["Sasrec_model_descritption"]["hidden_units"]])
        model.load_weights(weights_path)
        logging.info(f"Loaded Tensorflow model weights from the path {weights_path}")

        ##Item ids start from 1 and end with total num item +1
        candidates = np.array([[i for i in  range(1, model.item_num+1)]])

        return item_id_for_extid, item_for_cf_id, model, candidates

    item_id_for_extid, extid_for_itemid, model, candidates = get_model_and_meta.__func__()
 
    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them."""
        model = cls.model
        sequence = input["history"]
        n = input["n_items"]

        # Convert all the external ids to item ids
        sequence = [cls.item_id_for_extid[item]
            for item in sequence
            if item in cls.item_id_for_extid]
        
        # If the length of the user sequence is greater than max_seq_length, we only consider the recent interactions (i.e) last interactions.
        # else consider all the items that user has interacted. (i.e) the same sequence
        sequence = sequence[-model.seq_max_len:] if len(sequence) >= model.seq_max_len else sequence

        if not sequence:
            err = "No sequence given for predictions/recommendations."
            logging.error(err)
            return []
        
        #Padding the sequences
        input_sequence = np.array([np.pad(sequence, (model.seq_max_len-len(sequence),0), 'constant')])

        #Input sequnces, with the candidature with toal items array
        input_predictions = {'input_seq': input_sequence, 'candidate': cls.candidates}

        #Predictions for model history
        predictions_for_items = model.predict(input_predictions)

        #Convert into numpy
        converted_predictions = predictions_for_items.numpy().flatten()

        #Enumerate the list starting form index 1 as item id's as start from 1 during pre-processing
        enumerated_list = enumerate(converted_predictions, start=1)

        #Sort the predictions in descending order
        predictions_sorted = sorted(enumerated_list, key =itemgetter(1), reverse=True)

        #Obtain only top n recommendations
        if n is not None:
            predictions_sorted = islice(predictions_sorted, n)
        
        #Normalize the predictions
        predictions_sorted = normalize_scores(predictions_sorted)

        return [ (cls.extid_for_itemid[item[0]], item[1])
            for item in predictions_sorted
            if item[0] in cls.extid_for_itemid]

        
    @classmethod
    def predict_batch_eval(cls, padded_sequences):

        """For the input batch of sequences, do the model predictions"""
        model = cls.model

        #Input sequnces, with the candidature with toal items array
        input_predictions = {'input_seq': padded_sequences, 'candidate': cls.candidates}

         #Predictions for model history
        predictions_for_item_ids = model.predict(input_predictions)

        return predictions_for_item_ids


# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ModelService.model is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response="\n", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    if flask.request.content_type != "application/json":
        # this service only handles JSON
        return flask.Response(
            response="This predictor only supports JSON data",
            status=415,
            mimetype="text/plain",
        )

    # Convert from CSV to pandas
    data = json.loads(flask.request.data)
    # print(data)

    # Do the prediction
    predictions = ModelService.predict(data)
    result = {"predictions": predictions}
    # create a compact representation without indentation, newlines and whitespaces
    result = json.dumps(result, indent=None, separators=(',', ':'))
    return flask.Response(response=result, status=200, mimetype="application/json")
