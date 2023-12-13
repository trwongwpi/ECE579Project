import torch
from torch import nn
import torch.nn.functional as F
from tool.torch_utils import *
from models import Yolov4
from train import evaluate, train
from cfg import Cfg
from dataset import Yolo_dataset
from torch.utils.data import DataLoader
from tool.tv_reference.utils import collate_fn as val_collate


from collections import namedtuple

Codebook = namedtuple('Codebook', ['centroids', 'labels'])

def fine_grained_prune(tensor: torch.Tensor, sparsity : float) -> torch.Tensor:
    """
    magnitude-based pruning for single tensor
    :param tensor: torch.(cuda.)Tensor, weight of conv/fc layer
    :param sparsity: float, pruning sparsity
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    :return:
        torch.(cuda.)Tensor, mask for zeros
    """
    sparsity = min(max(0.0, sparsity), 1.0)
    if sparsity == 1.0:
        tensor.zero_()
        return torch.zeros_like(tensor)
    elif sparsity == 0.0:
        return torch.ones_like(tensor)

    num_elements = tensor.numel()
    ##################### YOUR CODE STARTS HERE #####################
    # Step 1: calculate the #zeros (please use round())
    num_zeros = round(num_elements * sparsity)
    # Step 2: calculate the importance of weight
    importance = torch.abs(tensor)
    # Step 3: calculate the pruning threshold
    threshold = torch.kthvalue(importance.view(-1), num_zeros).values
    # print(threshold[0])
    # Step 4: get binary mask (1 for nonzeros, 0 for zeros)
    mask = torch.gt(importance, threshold)
    # print(mask)
    ##################### YOUR CODE ENDS HERE #######################

    # Step 5: apply mask to prune the tensor
    tensor.mul_(mask)

    return mask


def calculate_very_slow_exponential_sparsity(layer_index, max_sparsity):
    # Use a very slow exponential increase by dividing the layer_index by a larger constant
    sparsity = max_sparsity * (1 - (1e-7 ** (layer_index / (327 * 10))))
    
    # Clip the sparsity value to ensure it is within the range [0, 0.9]
    sparsity = max(0, min(0.9, sparsity))
    
    return sparsity



class FineGrainedPruner:
    def __init__(self, model, sparsity_dict):
        self.masks = FineGrainedPruner.prune(model, sparsity_dict)

    @torch.no_grad()
    def apply(self, model):
        for name, param in model.named_parameters():
            if name in self.masks:
                param *= self.masks[name]

    @staticmethod
    @torch.no_grad()
    def prune(model, sparsity_dict):
        masks = dict()
        for name, param in model.named_parameters():
            if param.dim() > 1: # we only prune conv and fc weights
                masks[name] = fine_grained_prune(param, sparsity_dict[name])
        return masks
    

def get_num_parameters(model: nn.Module, count_nonzero_only=False) -> int:
    """
    calculate the total number of parameters of model
    :param count_nonzero_only: only count nonzero weights
    """
    num_counted_elements = 0
    for param in model.parameters():
        if count_nonzero_only:
            num_counted_elements += param.count_nonzero()
        else:
            num_counted_elements += param.numel()
    return num_counted_elements


def get_model_size(model: nn.Module, data_width=32, count_nonzero_only=False) -> int:
    """
    calculate the model size in bits
    :param data_width: #bits per element
    :param count_nonzero_only: only count nonzero weights
    """
    return get_num_parameters(model, count_nonzero_only) * data_width

def update_codebook(fp32_tensor: torch.Tensor, codebook: Codebook):
    """
    update the centroids in the codebook using updated fp32_tensor
    :param fp32_tensor: [torch.(cuda.)Tensor]
    :param codebook: [Codebook] (the cluster centroids, the cluster label tensor)
    """
    n_clusters = codebook.centroids.numel()
    fp32_tensor = fp32_tensor.view(-1)
    for k in range(n_clusters):
    ############### YOUR CODE STARTS HERE ###############
        # hint: one line of code
        mask = codebook.labels == k
        codebook.centroids[k] = fp32_tensor[mask].mean()
    ############### YOUR CODE ENDS HERE #################
from fast_pytorch_kmeans import KMeans

def k_means_quantize(fp32_tensor: torch.Tensor, bitwidth=4, codebook=None):
    """
    quantize tensor using k-means clustering
    :param fp32_tensor:
    :param bitwidth: [int] quantization bit width, default=4
    :param codebook: [Codebook] (the cluster centroids, the cluster label tensor)
    :return:
        [Codebook = (centroids, labels)]
            centroids: [torch.(cuda.)FloatTensor] the cluster centroids
            labels: [torch.(cuda.)LongTensor] cluster label tensor
    """
    if codebook is None:
        ############### YOUR CODE STARTS HERE ###############
        # get number of clusters based on the quantization precision
        # hint: one line of code
        n_clusters =2**bitwidth
        ############### YOUR CODE ENDS HERE #################
        # use k-means to get the quantization centroids
        kmeans = KMeans(n_clusters=n_clusters, mode='euclidean', verbose=0)
        labels = kmeans.fit_predict(fp32_tensor.view(-1, 1)).to(torch.long)
        centroids = kmeans.centroids.to(torch.float).view(-1)
        codebook = Codebook(centroids, labels)
    ############### YOUR CODE STARTS HERE ###############
    # decode the codebook into k-means quantized tensor for inference
    # hint: one line of code
    quantized_tensor = codebook[0][codebook[1]].view(fp32_tensor.shape)
    ############### YOUR CODE ENDS HERE #################
    fp32_tensor.set_(quantized_tensor.view_as(fp32_tensor))
    return codebook

from torch.nn import parameter
class KMeansQuantizer:
    def __init__(self, model : nn.Module, bitwidth=4):
        self.codebook = KMeansQuantizer.quantize(model, bitwidth)

    @torch.no_grad()
    def apply(self, model, update_centroids):
        for name, param in model.named_parameters():
            if name in self.codebook:
                if update_centroids:
                    update_codebook(param, codebook=self.codebook[name])
                self.codebook[name] = k_means_quantize(
                    param, codebook=self.codebook[name])

    @staticmethod
    @torch.no_grad()
    def quantize(model: nn.Module, bitwidth=4):
        codebook = dict()
        if isinstance(bitwidth, dict):
            for name, param in model.named_parameters():
                if name in bitwidth:
                    codebook[name] = k_means_quantize(param, bitwidth=bitwidth[name])
        else:
            for name, param in model.named_parameters():
                if param.dim() > 1:
                    codebook[name] = k_means_quantize(param, bitwidth=bitwidth)
        return codebook
    


if __name__ == "__main__":
    Byte = 8
    KiB = 1024 * Byte
    MiB = 1024 * KiB
    GiB = 1024 * MiB
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Yolov4(yolov4conv137weight=None, n_classes=52, inference=True)

    pretrained_dict = torch.load(r'G:\sources\pytorch-YOLOv4\Yolov4_epoch10.pth', map_location=torch.device('cuda'))
    model.load_state_dict(pretrained_dict)

    use_cuda = True
    if use_cuda:
        model.cuda()


    max_sparsity_value = 0.9  # You want the sparsity values to range from 0 to 0.9
    bitwidth = 8
    sparsity_values = {}

    for index ,(name, param) in enumerate(model.named_parameters()):
        layer_name = name
        sparsity = calculate_very_slow_exponential_sparsity(index, max_sparsity_value)
        sparsity_values[layer_name] = sparsity

    quantizer = KMeansQuantizer(model, bitwidth)
    quantizer.apply(model, update_centroids=False)

    pruner = FineGrainedPruner(model, sparsity_values)

    sparse_model_size = get_model_size(model, count_nonzero_only=True)

    Cfg.test_label = r'G:\sources\pytorch-YOLOv4\data\roboflow\test\_annotations.txt'
    Cfg.dataset_dir = r'G:\sources\pytorch-YOLOv4\data\roboflow\test'

    train(model, device, Cfg, epochs=5, batch_size=1, save_cp=True, log_step=20, img_scale=0.5, callbacks=[lambda: pruner.apply(model), lambda: quantizer.apply(model)] )

    val_dataset = Yolo_dataset(Cfg.test_label, Cfg, train=False)
    val_loader = DataLoader(val_dataset, batch_size=Cfg.batch // Cfg.subdivisions, shuffle=True, num_workers=8,
                        pin_memory=True, drop_last=True, collate_fn=val_collate)
    

    print('evaluating...')
    evaluator = evaluate(model, val_loader, Cfg, device)



    torch.save(model.state_dict(), r'G:\sources\pytorch-YOLOv4\sparsed.pth')

    stats = evaluator.coco_eval['bbox'].stats

    print(f"AP: {stats[0]}")
    print(f"AP50: {stats[1]}")
    print(f"AP75: {stats[2]}")
    print(f"AP_small: {stats[3]}")
    print(f"AP_medium: {stats[4]}")
    print(f"AP_large: {stats[5]}")
    print(f"AR1: {stats[6]}")
    print(f"AR10: {stats[7]}")
    print(f"AR100: {stats[8]}")
    print(f"AR_small: {stats[9]}")
    print(f"AR_medium: {stats[10]}")
    print(f"AR_large: {stats[11]}")
    print(f"Sparse model has size={sparse_model_size / MiB:.2f} MiB")
    

