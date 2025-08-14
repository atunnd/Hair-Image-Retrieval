import torch
import random
import numpy as np
from collections import defaultdict, Counter
import faiss
import os


class Kmean_Faiss:
    def __init__(self, dim=128, k=15, device=0, device_id=0, momentum=0.9, save_path=""):
        self.dim = dim
        self.k = k
        self.device = device
        self.device_id = device_id
        self.momentum = momentum  
        self.res = faiss.StandardGpuResources()

        self.kmeans = faiss.Clustering(dim, k)
        self.kmeans.niter = 25
        self.kmeans.verbose = False

        self.index_flat = faiss.IndexFlatL2(dim)
        #self.index_flat = faiss.index_cpu_to_gpu(self.res, device_id, self.index_flat)

        self.save_path = os.path.join(save_path, "centroids")
        os.makedirs(self.save_path, exist_ok=True)


    def fit(self, embeddings: torch.Tensor, batch_id: int):
        # Convert to numpy for FAISS
        emb_np = embeddings.detach().cpu().numpy().astype("float32")
        self.kmeans.train(emb_np, self.index_flat)

        # Get centroids from FAISS and store them as torch tensor on GPU
        centroids_np = faiss.vector_float_to_array(self.kmeans.centroids) \
            .reshape(self.k, self.dim).astype("float32")

        self.centroids = torch.from_numpy(centroids_np).to(embeddings.device)

        # Update centroid with EMA
        centroid_path = os.path.join(self.save_path, f"centroid_{batch_id}")
        if os.path.exists(centroid_path):
            loading_path = centroid_path
            ema_centroids = torch.load(loading_path, map_location=self.device)
            self.centroids = self.momentum*ema_centroids + (1-self.momentum) *self.centroids
        
        # save new centroids
        torch.save(self.centroids, centroid_path)

    def query_hard_negative_centroid(self, embeddings: torch.Tensor, query_id: int):
        # Compute distances from the query sample to all centroids → pick top 2
        query_vec = embeddings[query_id].unsqueeze(0)  # shape: (1, dim)
        distances = torch.cdist(query_vec, self.centroids)  # shape: (1, k)
        top2_idx = torch.topk(distances, k=2, dim=1).indices
        return self.centroids[top2_idx[:, 1]]  # return the 2nd closest centroid

    def query_hard_negative_sample(self, embeddings: torch.Tensor, query_id: int):
        # Get the 2nd closest centroid
        negative_centroid = self.query_hard_negative_centroid(embeddings, query_id).unsqueeze(0)

        # Remove the query sample from the embeddings
        mask = torch.ones(embeddings.size(0), dtype=torch.bool, device=embeddings.device)
        mask[query_id] = False
        embeddings_filtered = embeddings[mask]

        # Find the farthest sample from the 2nd centroid
        distances = torch.cdist(negative_centroid, embeddings_filtered)  # shape: (1, N-1)
        #print("Distances: ", distances.shape)
        negative_idx = torch.argmax(distances, dim=2)
        return embeddings_filtered[negative_idx]

class NegSamplerMiniBatch(torch.nn.Module):
    def __init__(self, k=5, dim=128, momentum=0.9, device="cuda", device_id=0, negative_centroid=True, save_path=""):
        super(NegSamplerMiniBatch, self).__init__()
        self.k = k
        self.dim = dim
        self.kmeans = Kmean_Faiss(dim, k, device, device_id, momentum, save_path)
        self.negative_centroid = negative_centroid
        self.query = self.kmeans.query_hard_negative_centroid if self.negative_centroid else self.kmeans.query_hard_negative_sample

    def forward(self, embeddings, batch_id):
        """
        batch: batch embeddings
        Return: hard negative sample/centroid
        """

        self.kmeans.fit(embeddings, batch_id)

        neg_embeddings = torch.zeros_like(embeddings) 
        for idx in range(len(embeddings)):
            neg_embeddings[idx] = self.query(embeddings, idx)
        
        return neg_embeddings

def NegSamplerClasses():
    pass

def NegSamplerRandomly(embeddings: torch.Tensor):
    """
    Randomly shuffle all samples in the batch embeddings.
    
    Args:
        embeddings (torch.Tensor): shape (batch_size, dim)
        
    Returns:
        shuffled_embeddings (torch.Tensor): shuffled embeddings
        permutation (torch.Tensor): indices of shuffle (to map original -> shuffled)
    """
    batch_size = embeddings.size(0)
    permutation = torch.randperm(batch_size, device=embeddings.device)
    shuffled_embeddings = embeddings[permutation]
    return shuffled_embeddings