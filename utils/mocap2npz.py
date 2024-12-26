import os
import json
import numpy as np

# 把json文件转化为npz文件

def load_json_files(json_dir):
    data0 = {
        "Rh": [],
        "Th": [],
        "poses": [],
        "expression": [],
        "shapes": []
    }
    data = {
        "global_orient": [],
        "transl": [],
        "body_pose": [],
        "jaw_pose": [], 
        "betas": [],
        "expression": [],
        "left_hand_pose": [],
        "right_hand_pose": []
    }

    # Iterate through all JSON files in the directory
    for file_name in sorted(os.listdir(json_dir)):
        if file_name.endswith('.json'):
            file_path = os.path.join(json_dir, file_name)
            with open(file_path, 'r') as json_file:
                json_data = json.load(json_file)
                if json_data:
                    entry = json_data[0]  # Assuming there's only one entry per file
                    data0["Rh"].append(entry["Rh"])
                    # data0["Th"].append(entry["Th"] - np.array([-.75, -.5, 2.]))
                    data0["Th"].append(entry["Th"])
                    data0["poses"].append(entry["poses"])
                    # data0["left_hand_pose"].append(entry["poses"][75:120])
                    # data0["right_hand_pose"].append(entry["poses"][121:])
                    data0["shapes"].append(entry["shapes"])
                    data0["expression"].append(entry["expression"])

    # Convert lists to numpy arrays
    data["global_orient"] = np.concatenate(data0["Rh"], axis=0)
    data["transl"] = np.concatenate(data0["Th"], axis=0)
    # data["global_orient"] = np.zeros((len(data0["Rh"]),3))
    # data["transl"] = np.zeros((len(data0["Th"]),3))
    data["body_pose"] = np.concatenate(data0["poses"], axis=0)
    data["jaw_pose"] = data["body_pose"][:, 66:69]
    data["left_hand_pose"] = data["body_pose"][:, 75:120]
    data["right_hand_pose"] = data["body_pose"][:, 120:]
    data["body_pose"] = data["body_pose"][:, 3:66] # may change
    data["expression"] = np.concatenate(data0["expression"], axis=0)
    data["betas"] = np.concatenate(data0["shapes"][:1], axis=0)
    # data["betas"] = np.concatenate(data0["shapes"], axis=0)
    # data["betas"] = np.concatenate(data0["shapes"], axis=0)

    return data

def save_to_npz(data, output_file):
    np.savez(output_file, **data)

if __name__ == "__main__":
    json_dir = '/data1/hlp/dataset/241129/human/output/smplx/smpl_full'
    output_file = '/nas_data/home/hlp/data/angs/zlb/smpl_params.npz'
    
    data = load_json_files(json_dir)
    print(len(data["betas"]))
    save_to_npz(data, output_file)
    print(f"Data has been saved to {output_file}")