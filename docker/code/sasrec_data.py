# Copyright (c) 2024, ZDF.
class CustomSASRecDataSet():
    """Generates the count of users and items and creates a dict of users to their items."""

    def __init__(self, data):

        self.data = data
        self.usernum = max(self.data["user_id"])
        self.itemnum = max(self.data['item_id'])
        self.user_data = self.data.groupby("user_id")["item_id"].apply(list).to_dict()





