import torch


# type_of_triplets: options are "all", "hard", or "semihard".
#                 "all" means all triplets that violate the margin
#                 "hard" is a subset of "all", but the negative is closer to the anchor than the positive
#                 "semihard" is a subset of "all", but the negative is further from the anchor than the positive
#             "easy" is all triplets that are not in "all"
class TripletMiner:
    def __init__(self, margin=0.2, type_of_triplets='all', sim_flag = False):
        self.margin = margin
        self.type_of_triplets = type_of_triplets
        self.sim_flag = sim_flag

    def get_all_trip_indices(self, labels):
        labels1 = labels.unsqueeze(1)
        labels2 = labels.unsqueeze(0)
        matches = (labels1 == labels2).byte()
        diffs = matches ^ 1
        matches.fill_diagonal_(0)
        triplets = matches.unsqueeze(2) * diffs.unsqueeze(1)
        return torch.where(triplets)

    def mine(self, X_embed, y, return_dist = False):
        anchor_idx, positive_idx, negative_idx = self.get_all_trip_indices(y)
        if (self.sim_flag):
            mat = torch.matmul(X_embed, X_embed.t())
        else:
            mat = torch.cdist(X_embed, X_embed)
        ap_dist = mat[anchor_idx, positive_idx]
        an_dist = mat[anchor_idx, negative_idx]
        triplet_margin = (
            ap_dist - an_dist if self.sim_flag else an_dist - ap_dist
        )
        if self.type_of_triplets == "easy":
            threshold_condition = triplet_margin > self.margin
        else:
            threshold_condition = triplet_margin <= self.margin
            if self.type_of_triplets == "hard":
                threshold_condition &= triplet_margin <= 0
            elif self.type_of_triplets == "semihard":
                threshold_condition &= triplet_margin > 0

        trip_ind = (anchor_idx[threshold_condition], positive_idx[threshold_condition],
                    negative_idx[threshold_condition])
        if (return_dist):
            return trip_ind, mat
        else:
            return trip_ind








