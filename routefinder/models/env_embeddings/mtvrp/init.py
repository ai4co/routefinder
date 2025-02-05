import torch
import torch.nn as nn


class MTVRPInitEmbedding(nn.Module):
    """Initial embedding MTVRP.

    Note: this is the same as what MTPOMO and MVMoE use.

    Customer features:
        - locs: x, y euclidean coordinates
        - demand_linehaul: demand of the nodes (delivery) (C)
        - demand_backhaul: demand of the nodes (pickup) (B)
        - time_windows: time window (TW)
        - durations: duration of the nodes (TW)

    Global features:
        - loc: x, y euclidean coordinates of depot
    """

    def __init__(
        self, embed_dim=128, bias=False, num_global_feats=2, num_cust_feats=7, **kw
    ):  # node: linear bias should be false in order not to influence the embedding if
        super(MTVRPInitEmbedding, self).__init__()

        # Depot feats (includes global features): x, y, distance, backhaul_class, open_route
        global_feat_dim = num_global_feats
        self.project_global_feats = nn.Linear(global_feat_dim, embed_dim, bias=bias)

        # Customer feats: x, y, demand_linehaul, demand_backhaul, time_window_early, time_window_late, durations
        customer_feat_dim = num_cust_feats
        self.project_customers_feats = nn.Linear(customer_feat_dim, embed_dim, bias=bias)

        self.embed_dim = embed_dim

    def forward(self, td):
        # Global (batch, 1, 2) -> (batch, 1, embed_dim)
        global_feats = td["locs"][:, :1, :]

        # Customers (batch, N, 5) -> (batch, N, embed_dim)
        # note that these feats include the depot (but unused) so we exclude the first node
        cust_feats = torch.cat(
            # TODO replace 1 with actual number of depots
            (
                td["demand_linehaul"][..., 1:, None],
                td["demand_backhaul"][..., 1:, None],
                td["time_windows"][..., 1:, :],
                td["service_time"][..., 1:, None],
                td["locs"][:, 1:, :],
            ),
            -1,
        )

        # If some features are infinity (e.g. distance limit is inf because of no limit), replace with 0 so that it does not affect the embedding
        global_feats = torch.nan_to_num(global_feats, nan=0.0, posinf=0.0, neginf=0.0)
        cust_feats = torch.nan_to_num(cust_feats, nan=0.0, posinf=0.0, neginf=0.0)
        global_embeddings = self.project_global_feats(
            global_feats
        )  # [batch, 1, embed_dim]
        cust_embeddings = self.project_customers_feats(
            cust_feats
        )  # [batch, N, embed_dim]
        return torch.cat(
            (global_embeddings, cust_embeddings), -2
        )  # [batch, N+1, embed_dim]


# Note that this is the most recent version and should be used from now on. Others can be based on this!
class MTVRPInitEmbeddingRouteFinderBase(nn.Module):
    """General Init embedding class

    Args:
        num_global_feats: number of global features
        num_cust_feats: number of customer features
        embed_dim: embedding dimension
        bias: whether to use bias in the linear layers
        posinf_val: value to replace positive infinity values with
    """

    def __init__(
        self, num_global_feats, num_cust_feats, embed_dim=128, bias=False, posinf_val=0.0
    ):
        super(MTVRPInitEmbeddingRouteFinderBase, self).__init__()
        self.project_global_feats = nn.Linear(num_global_feats, embed_dim, bias=bias)
        self.project_customers_feats = nn.Linear(num_cust_feats, embed_dim, bias=bias)
        self.embed_dim = embed_dim
        self.posinf_val = posinf_val

    def _global_feats(self, td):
        raise NotImplementedError("This method should be overridden by subclasses")

    def _cust_feats(self, td):
        raise NotImplementedError("This method should be overridden by subclasses")

    def forward(self, td):
        global_feats = self._global_feats(td)
        cust_feats = self._cust_feats(td)

        global_feats = torch.nan_to_num(global_feats, posinf=self.posinf_val)
        cust_feats = torch.nan_to_num(cust_feats, posinf=self.posinf_val)
        global_embeddings = self.project_global_feats(global_feats)
        cust_embeddings = self.project_customers_feats(cust_feats)

        return torch.cat((global_embeddings, cust_embeddings), -2)


class MTVRPInitEmbeddingRouteFinder(MTVRPInitEmbeddingRouteFinderBase):
    """
    Customer features:
        - locs: x, y euclidean coordinates
        - demand_linehaul: demand of the nodes (delivery) (C)
        - demand_backhaul: demand of the nodes (pickup) (B)
        - time_windows: time window (TW)
        - service_time: service time of the nodes
    Global features:
        - open_route (O)
        - distance_limit (L)
        - (end) time window of depot
        - x, y euclidean coordinates of depot
    The above features are embedded in the depot node as global and get broadcasted via attention.
    This allows the network to learn the relationships between them.
    """

    def __init__(self, embed_dim=128, bias=False, posinf_val=0.0):
        super(MTVRPInitEmbeddingRouteFinder, self).__init__(
            num_global_feats=5,  # x, y, open_route, distance_limit, time_window_depot
            num_cust_feats=7,
            embed_dim=embed_dim,
            bias=bias,
            posinf_val=posinf_val,
        )

    def _global_feats(self, td):
        return torch.cat(
            [
                td["open_route"].float()[..., None],
                # TODO replace 1 with num_depots
                td["locs"][:, :1, :],
                td["distance_limit"][..., None],
                td["time_windows"][:, :1, 1:2],
            ],
            -1,
        )

    def _cust_feats(self, td):
        return torch.cat(
            (
                td["locs"][..., 1:, :],
                td["demand_linehaul"][..., 1:, None],
                td["demand_backhaul"][..., 1:, None],
                td["time_windows"][..., 1:, :],
                td["service_time"][..., 1:, None],
            ),
            -1,
        )


class MTVRPInitEmbeddingM(MTVRPInitEmbeddingRouteFinder):
    def __init__(self, embed_dim=128, bias=False, posinf_val=0.0):
        # Note: here we add the backhaul_class as a feature
        MTVRPInitEmbeddingRouteFinderBase.__init__(
            self,
            num_global_feats=5 + 1,
            num_cust_feats=7,
            embed_dim=embed_dim,
            bias=bias,
            posinf_val=posinf_val,
        )

    def _global_feats(self, td):
        glob_feats = super(MTVRPInitEmbeddingM, self)._global_feats(td)
        is_mixed_backhaul = (td["backhaul_class"] == 2).float()
        return torch.cat([glob_feats, is_mixed_backhaul[..., None]], -1)


class MTVRPInitEmbeddingFull(MTVRPInitEmbeddingRouteFinder):
    """
    This is the full embedding, including the Mixed Backhaul (MB) variants
    and Multi-depot (MD) variants as well.
    Node features:
        Customer features:
            - locs: x, y euclidean coordinates
            - demand_linehaul: demand of the nodes (delivery) (C)
            - demand_backhaul: demand of the nodes (pickup) (B)
            - time_windows: time window (TW)
            - service_time: service time of the nodes
        Depot features:
            - (end) time window of depots
            - x, y euclidean coordinates of depots (MD, optional)
    Global features:
        - open_route (O)
        - distance_limit (L)
        - backhaul_class (MB)
    This allows the network to learn the relationships between them.
    Note that the global features in practice are embedded in the depots, since
    they represent vehicle characteristics.
    """

    def __init__(self, embed_dim=128, bias=False, posinf_val=0.0):
        super(MTVRPInitEmbeddingRouteFinder, self).__init__(
            num_global_feats=6,  # x, y, open_route, distance_limit, time_window_depot, backhaul_class
            num_cust_feats=7,
            embed_dim=embed_dim,
            bias=bias,
            posinf_val=posinf_val,
        )

    def _global_feats(self, td):
        """Global features include the depot node(s) features"""
        num_depots = td["num_depots"].max().item()
        return torch.cat(
            [
                td["open_route"].float()[..., None].repeat(1, num_depots, 1),
                td["locs"][:, :num_depots, :],
                td["distance_limit"][..., None].repeat(1, num_depots, 1),
                td["time_windows"][:, :num_depots, 1:2],
                (td["backhaul_class"] == 2).float()[..., None].repeat(1, num_depots, 1),
            ],
            -1,
        )

    def _cust_feats(self, td):
        num_depots = td["num_depots"].max().item()
        return torch.cat(
            (
                td["locs"][..., num_depots:, :],
                td["demand_linehaul"][..., num_depots:, None],
                td["demand_backhaul"][..., num_depots:, None],
                td["time_windows"][..., num_depots:, :],
                td["service_time"][..., num_depots:, None],
            ),
            -1,
        )
