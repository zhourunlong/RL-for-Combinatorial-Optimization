import os
import torch

def simplify_path(path):
    s_list = path.split("/")
    temp_str = " ".join(s_list)
    s_list = temp_str.split()
    
    new_list = []
    
    for item in s_list:
        if item == ".":
            continue
            
        elif item == "..":
            if new_list:
                new_list.pop(-1)
                
        else:
            new_list.append(item)
            
    new_str = "/".join(new_list)
    if path[-1] == "/":
        new_str += "/"
    
    return new_str

def get_file_number(dir):
    _, fn = os.path.split(dir)
    num, _ = os.path.splitext(fn)
    num = "x" + num
    for i in range(len(num) - 1, -1, -1):
        if not num[i].isdigit():
            return int(num[i+1:])

def copy_logs(fn, logger, smooth_episode):
    log_package = torch.load(fn, map_location="cpu")
    for key, val in log_package.items():
        if "#sample" in key:
            sample_cnts = val
            break

    for key, val in log_package.items():
        if "episode" in key or "#sample" in key:
            continue
        save_episode = val.shape[0]
        for i in range(0, save_episode, smooth_episode):
            logger.log_stat(key, val[i:i+smooth_episode].mean(), int(sample_cnts[i+smooth_episode-1]))
    
    for key, val in log_package.items():
        if not "episode" in key:
            continue
        for i in range(0, save_episode, smooth_episode):
            logger.log_stat(key, val - save_episode + i + smooth_episode, int(sample_cnts[i+smooth_episode-1]))
    
    return int(sample_cnts[-1]) # returns the sample count