hp_cora = {
    "wd1": 4e-2,
    "wd2": 2e-5,
    "lambda_1": 2.8,
    "lambda_2": 0.2,
    "lambda_3": 0.1,
    "layer": 2,
    "dropout": 0.35,
    "lr": 0.02,
    "alpha": 0.8,
}   # 8875 --9069   --9088

hp_pubmed = {
    "wd1": 4e-2,
    "wd2": 2e-5,
    "lambda_1": 3.2,
    "lambda_2": 1.8,
    "lambda_3": 0.1,
    "layer": 2,
    "dropout": 0.35,
    "lr": 0.03,
    "alpha": 0.8,
}   # 8813(9005)

hp_citeseer = {
    "wd1": 4e-2,
    "wd2": 5e-5,
    "lambda_1": 2.0,
    "lambda_2": 1.2,
    "lambda_3": 0.15,
    "layer": 2,
    "dropout": 0.3, #0.3
    "lr": 0.015,
    "alpha": 0.8,
}   # 76.70 --7779 (lower than 7799)  --7735

hp_chameleon = {
    "wd1": 2e-4,
    "wd2": 0.0,
    "lambda_1": 3.5,
    "lambda_2": 4.0,
    "lambda_3": 0.01,
    "layer": 2,
    "dropout": 0.25,
    "lr": 0.02,
    "alpha": 0.2,
}   # 7463  -7778   --7843

hp_squirrel = {
    "wd1": 0.075,
    "wd2": 0.0,
    "lambda_1": 1.2,
    "lambda_2": 1.1,    #1.2(1.3) 6449
    "lambda_3": 0.01,
    "layer": 2,
    "dropout": 0.25,
    "lr": 0.02,
    "alpha": 0.3,
}   # 6141 --6411   --6478

hp_amazon_computers = {
    "wd1": 5e-4,
    "wd2": 0.0,
    "lambda_1": 0.5,
    "lambda_2": 1.5,
    "lambda_3": 0.2,
    "layer": 2,
    "dropout": 0.02,
    "lr": 0.03,
    "alpha": 0.85,
}   # 8960 --9062 --9076

hp_amazon_photo = {
    "wd1": 6e-4,
    "wd2": 0.0,
    "lambda_1": 2.0,
    "lambda_2": 1.4,
    "lambda_3": 0.15,
    "layer": 2,
    "dropout": 0.01,
    "lr": 0.05,
    "alpha": 0.9,
}   #9317 --9297  -- 9297

def get_hyper_param(name: str):
    name = name.lower()
    if name == "cora":
        return hp_cora
    elif name == "pubmed":
        return hp_pubmed
    elif name == "citeseer":
        return hp_citeseer
    elif name == "chameleon":
        return hp_chameleon
    elif name == "squirrel":
        return hp_squirrel
    elif name == "computers":
        return hp_amazon_computers
    elif name == "photo":
        return hp_amazon_photo
    else:
        raise Exception("Not available")
