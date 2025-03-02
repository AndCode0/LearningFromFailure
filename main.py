import wandb
import torch
import Models.learnedfromfailing2implement as lff

# TODO: ordina e commenta tutti i parametri
conf = dict(
    dataset_tag="CelebA",
    model_tag="ResNet18",
    optimizer_tag="Adam",
    data_dir="C:\\Users\\aconte\\Desktop",
    save_dir="C:\\Users\\aconte\\Desktop\\Weights",
    save_model=True,
    target_attr_idx=9,  # BlondHair for CelebA
    bias_attr_idx=20,   # Male for CelebA
    batch_size=256,
    num_steps=10000,    # 636 * 200
    learning_rate=0.001,
    weight_decay=1e-4,
    gce_q=0.7,
    ema_alpha=0.9,
    valid_freq=100,
    log_freq=10,
    num_workers=0,
    device="cuda" if torch.cuda.is_available() else "cpu",
)
wandb.login(key='1178d63dc9b4985e3e239e7604325d6cf907a9b1')
with wandb.init(mode="offline", config=conf):
# with wandb.init(
#     entity="learning-from-failure", project="fullModel", id="debugRun", config=conf
# ) as run:
    model = lff.LfFTrainer(wandb.config)
    results = model.train()  

wandb.finish()
