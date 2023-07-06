# You can also add multiple concepts here. Try tweaking `--max_train_steps` accordingly.

concepts_list = [
    {
        "instance_prompt":      "photo of student activitiy in school", # the general prompt for all image in my dataset
        "class_prompt":         "school",                               # the class name of them
        "instance_data_dir":    "/content/school_concept",              # path to folder
        "class_data_dir":       "/content/data/school"                  
    },
# if dataset have more category (image), append the concepts list
#     {
#         "instance_prompt":      "photo of ukj person",
#         "class_prompt":         "photo of a person",
#         "instance_data_dir":    "/content/data/ukj",
#         "class_data_dir":       "/content/data/person"
#     }
]

# `class_data_dir` contains regularization images
import json
import os
for c in concepts_list:
    os.makedirs(c["instance_data_dir"], exist_ok=True)

with open("concepts_list.json", "w") as f:
    json.dump(concepts_list, f, indent=4)