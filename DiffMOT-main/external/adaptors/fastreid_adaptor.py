import torch
import torchreid

class FastReID:
    def __init__(self, weight_path):
        self.model = torchreid.models.build_model(
            name="osnet_ain_x1_0",
            num_classes=1000,
            pretrained=False
        )

        ckpt = torch.load(weight_path, map_location="cpu")

        # safe loading (different repos use different keys)
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        self.model.load_state_dict(state_dict, strict=False)

        self.model.eval()
        self.model.cuda()

    def __call__(self, images):
        with torch.no_grad():
            return self.model(images)