from unirna.config import build_config

path = "/home/wangxi/project/Uni-RNA/unirna_fineturn/model_weight/unirna_L16_E1024_DPRNA500M_STEP400K.pt"
config = build_config(path)
config.save_pretrained("config.json")
