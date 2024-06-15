import numpy as np
import torch
import torch.nn.functional as F


def cosine_similarity(v1, v2):
    # Convert lists to numpy arrays
    v1 = np.array(v1)
    v2 = np.array(v2)

    # Compute the cosine similarity
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    similarity = dot_product / (norm_v1 * norm_v2)
    return similarity

def DSI_metric(weight_history):
    # weight_history in shape [layers, modules, dims]
    import torch
    import torch.nn.functional as F


def get_top_k_modules(DSI, top_k):
    """
    Get the indices of the top-k modules based on DSI values.
    DSI: torch.Tensor - shape [layers, module_numbers]
    top_k: int - number of top modules to return
    """
    # Flatten the DSI to work with one-dimensional sorting
    flat_dsi = DSI.view(-1) * -1

    # Get the top-k values and their indices in the flattened DSI
    _, indices = torch.topk(flat_dsi, top_k)

    # Convert flat indices to 2D indices
    top_k_indices = [(index // DSI.size(1), index % DSI.size(1)) for index in indices]

    return top_k_indices

def calculate_DSI(weights):
    weights = weights.permute(1, 2, 3, 0)
    """
    Calculate the Dimension Stability Indicator (DSI) for each module in each layer.
    weights: torch.Tensor - shape [layers, module_numbers, candidates, K]
    """
    layers, module_numbers, candidates, K = weights.shape

    # Initialize the DSI array
    DSI = torch.zeros((layers, module_numbers))

    for layer in range(layers):
        for module in range(module_numbers):
            module_weights = weights[layer, module]  # shape [candidates, K]

            # Calculate mean across the interval for each candidate
            mean_weights = module_weights.mean(dim=1)  # shape [candidates]

            # Calculate standard deviation for each candidate
            std_devs = module_weights.std(dim=1)  # shape [candidates]

            # Normalize each weight vector to form a probability distribution
            distributions = F.softmax(module_weights, dim=0)  # shape [candidates, K]

            # Compute KL divergence between the first and last distributions
            kl_div = F.kl_div(distributions[:, 0].log(), distributions[:, -1], reduction='sum')

            # Average of the standard deviations
            avg_std_dev = std_devs.mean()

            # Calculate DSI for this module
            DSI[layer, module] = avg_std_dev * kl_div

    return DSI


def recognize_module_weights_loc(name):
    # to map a module name to the weights
    sub_ = name.split(".")
    if 'prefix' in name:
        layer_id = sub_[1].split("_")[1]
        if 'up' not in name:
            return (None, False, False, False, False)
        return (int(layer_id), False, False, False, False)
    layer_id = None
    lora_location_dict = {
        'q': 0, 'k': 1, 'v': 2, 'o': 3, 'wi': 4, 'wo': 5
    }
    adapter_location_dict = {
        'DenseReluDense': 7, 'SelfAttention': 6, 'sadapter': 8, 'padapter': 9
    }
    encoder_binary_location_dict = {
        'q.bitfit': 0, 'k.bitfit': 1, 'v.bitfit': 2, 'original_module.o.bitfit': 3,
        'wi.bitfit': 6, 'wo.bitfit': 7,
        '0.layer_norm.bitfit': 4, '0.layer_norm.lnfit': 5,
        '1.layer_norm.bitfit': 8, '1.layer_norm.lnfit': 9,
    }
    decoder_binary_location_dict = {
        'q.bitfit': 0, 'k.bitfit': 1, 'v.bitfit': 2, 'original_module.o.bitfit': 3,
        'wi.bitfit': 6, 'wo.bitfit': 7,
        '0.layer_norm.bitfit': 4, '0.layer_norm.lnfit': 5,
        '1.layer_norm.bitfit': 10, '1.layer_norm.lnfit': 11,
        '2.layer_norm.bitfit': 8, '2.layer_norm.lnfit': 9,
    }
    if 'block' in name:
        layer_order = sub_.index("block")
        layer_id = int(sub_[layer_order + 1])
    loc = None
    encoder_flag = False
    matrix_flag = False
    final_layer_norm_flag = False
    if 'encoder' in name:
        encoder_flag = True
        
    if 'lora' in name or 'adapter' in name:
        matrix_flag = True
        if 'lora' in name:
            lora_locc = sub_.index("lora")
            loc_name = sub_[lora_locc-1]
            loc = lora_location_dict[loc_name]
        elif 'sadapter' in name:
            loc = adapter_location_dict['sadapter']
        elif 'padapter' in name:
            loc = adapter_location_dict['padapter']
        else:
            lora_locc = sub_.index("adapter")
            loc_name = sub_[lora_locc - 1]
            loc = adapter_location_dict[loc_name]
        if 'decoder' in name:
            loc = None
    elif "final_layer_norm" not in name and ('bitfit' in name or 'lnfit' in name):
        name_dict = encoder_binary_location_dict if encoder_flag else decoder_binary_location_dict
        for k in name_dict:
            if k in name:
                loc = name_dict[k]
                break
    elif "final_layer_norm" in name:
        final_layer_norm_flag = True
        if "encoder" in name:
            loc = 1
            if "bitfit" in name:
                loc = 0
        else:
            loc = 3
            if "bitfit" in name:
                loc = 2
    else:
        pass
    return (layer_id, loc, encoder_flag, matrix_flag, final_layer_norm_flag)


def recognize_layer_id(name):
    sub_ = name.split(".")
    if 'prefix' in name:
        layer_id = sub_[1].split("_")[1]
        if 'up' not in name:
            return (None, False, False, False, False)
        return (int(layer_id), False, False, False, False)
    layer_id = None
    encoder_flag = False
    lora_location_dict = {
        'q': 0, 'k': 1, 'v': 2, 'o':3, 'wi':4, 'wo':5
    }
    encoder_binary_location_dict = {
        'q.bitfit': 0, 'k.bitfit': 1, 'v.bitfit': 2, 'original_module.o.bitfit': 3,
        'wi.bitfit': 6, 'wo.bitfit': 7,
        '0.layer_norm.bitfit': 4, '0.layer_norm.lnfit': 5,
        '1.layer_norm.bitfit': 8, '1.layer_norm.lnfit': 9,
    }
    decoder_binary_location_dict = {
        'q.bitfit': 0, 'k.bitfit': 1, 'v.bitfit': 2, 'original_module.o.bitfit': 3,
        'wi.bitfit': 6, 'wo.bitfit': 7,
        '0.layer_norm.bitfit': 4, '0.layer_norm.lnfit': 5,
        '1.layer_norm.bitfit': 10, '1.layer_norm.lnfit': 11,
        '2.layer_norm.bitfit': 8, '2.layer_norm.lnfit': 9,
    }
    adapter_location_dict = {
        'DenseReluDense': 7, 'SelfAttention': 6, 'sadapter': 8, 'padapter': 9
    }
    final_layer_norm_flag = False
    if "final_layer_norm" in sub_:
        final_layer_norm_flag = True

    loc = None
    if 'lora' in name or 'adapter' in name:
        matrix_flag = True
        if 'lora' in name:
            lora_locc = sub_.index("lora")
            loc_name = sub_[lora_locc-1]
            loc = lora_location_dict[loc_name]
        elif 'sadapter' in name:
            loc = adapter_location_dict['sadapter']
        elif 'padapter' in name:
            loc = adapter_location_dict['padapter']
        else:
            lora_locc = sub_.index("adapter")
            loc_name = sub_[lora_locc - 1]
            loc = adapter_location_dict[loc_name]
    else:
        matrix_flag = False
        if final_layer_norm_flag:
            if "encoder" in name:
                loc = 1
                if "bitfit" in name:
                    loc = 0
            else:
                loc = 3
                if "bitfit" in name:
                    loc = 2
        else:
            if "encoder" in name:
                for k in encoder_binary_location_dict:
                    if k in name:
                        loc = encoder_binary_location_dict[k]
                        break
            else:
                for k in decoder_binary_location_dict:
                    if k in name:
                        loc = decoder_binary_location_dict[k]
                        break

    if sub_[0] == "encoder":
        encoder_flag = True
    if not final_layer_norm_flag:
        layer_order = sub_.index("block")
        layer_id = int(sub_[layer_order + 1])

    return layer_id, encoder_flag, final_layer_norm_flag, matrix_flag, loc
