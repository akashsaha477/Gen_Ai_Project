"""P10 adaptive batch normalization: records are normalized on their own beats, independently, without changing the model."""

import numpy as np
import torch

from ecgaug.experiments import phase6_patient_adaptation as pa
from ecgaug.models.baselines import ShallowCNN


def _model():
    torch.manual_seed(0)
    model = ShallowCNN(n_classes=4, n_side=3, channels=(4, 8), kernel=3, side_hidden=4).eval()
    with torch.no_grad():  # training statistics far from the data, so adaptation must change the output
        for layer in model.modules():
            if isinstance(layer, torch.nn.BatchNorm1d):
                layer.running_mean.fill_(5.0)
                layer.running_var.fill_(9.0)
    return model


def test_adaptation_is_per_record_and_leaves_the_model_unchanged():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 32)).astype(np.float32)
    side = rng.normal(size=(40, 3)).astype(np.float32)
    records = np.array([1] * 20 + [2] * 20)
    model = _model()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    out = pa.adapted_predict(model, x, side, records)
    assert out.shape == (40, 4) and np.allclose(out.sum(axis=1), 1, atol=1e-5)
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())  # the frozen checkpoint is not touched
    plain = torch.softmax(model(torch.from_numpy(x), torch.from_numpy(side)), dim=1).detach().numpy()
    assert not np.allclose(out, plain, atol=1e-4)  # adaptation changes the prediction
    x2 = x.copy()
    x2[20:] += 3.0  # change record 2 only
    out2 = pa.adapted_predict(model, x2, side, records)
    assert np.allclose(out2[:20], out[:20], atol=1e-6)  # record 1 is adapted on its own beats only
