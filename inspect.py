import torch
import matplotlib.pyplot as plt


class delayed_call:
    def __init__(self, fn):
        self.fn = fn
        self.args = None
        self.kwargs = None
        self.parent = None
        self.polar = False

    def __call__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        return self

    def __getattr__(self, name):
        plt_attr = getattr(plt, name)
        if callable(plt_attr):
            chained = delayed_call(plt_attr)
            chained.parent = self
            return chained
        return plt_attr

    def set_polar(self):
        self.polar = True
        return self

    def run(self):
        if self.parent:
            self.parent.run()
        return self.fn(*self.args, **self.kwargs)


class lazy:
    def __getattr__(self, name):
        plt_attr = getattr(plt, name)
        if callable(plt_attr):
            return delayed_call(plt_attr)
        return plt_attr


L = lazy()


def inspect_linear(layer: torch.nn.Linear, input: torch.Tensor, name: str = "Linear"):
    weight = layer.weight
    bias = layer.bias

    input_norm = input.pow(2).sum(dim=2).sqrt()
    weight_norm = weight.pow(2).sum(dim=1).sqrt()

    print(weight.shape, weight.norm(dim=1).shape, weight.norm(dim=0).shape)
    print(input.shape, input.norm(dim=2).shape, input.norm(dim=1).shape)

    # calculate the activation, power and rho
    act = torch.matmul(input, weight.T)
    power = torch.matmul(input_norm.T, weight_norm.unsqueeze(dim=0))
    rho = act.squeeze(dim=0) / power

    # calculate the singular values of input, weight and activation
    ieig = torch.linalg.svdvals(input.squeeze(dim=0))
    weig = torch.linalg.svdvals(weight)
    aeig = torch.linalg.svdvals(act.squeeze(dim=0))

    # project weight to 2D and calculate the angle and norm
    U, S, Vh = torch.linalg.svd(weight)
    weight_2d = torch.matmul(weight, Vh.T[:, :2])
    weight_angles = torch.atan2(weight_2d[:, 1], weight_2d[:, 0])
    weight_norms = torch.norm(weight_2d, dim=1)

    # make a fake weight and calculate the activation, power and rho
    iweight = torch.normal(mean=0, std=0.02, size=weight.shape)
    iweight_norm = iweight.pow(2).sum(dim=1).sqrt()
    iweig = torch.linalg.svdvals(iweight)

    iact = torch.matmul(input, iweight.T)
    ipower = torch.matmul(input_norm.T, iweight_norm.unsqueeze(dim=0))
    irho = iact.squeeze(dim=0) / ipower

    plots = [
        [
            L.hist(input.flatten(), bins=100).title("i/ hist"),
            L.hist(input_norm.flatten(), bins=100).title("i/ token norm hist"),
            L.hist(input.pow(2).sum(dim=1).sqrt().flatten(), bins=100).title(
                "i/ channel norm hist"
            ),
            L.hist(ieig, bins=100).title("i/ eigs hist"),
        ],
        [
            L.hist(weight.flatten(), bins=100).title("w/ hist"),
            L.hist(weight_norm.flatten(), bins=100).title("w/ token norm hist."),
            L.hist(weight.pow(2).sum(dim=0).sqrt().flatten(), bins=100).title(
                "w/ channel norm hist."
            ),
            L.scatter(weight_angles.flatten(), weight_norms, 1)
            .yscale("log")
            .title("Weight Direction Distribution")
            .set_polar(),
        ],
        [
            L.hist(act.flatten(), bins=100).title("act hist"),
            L.hist(power.flatten(), bins=100).title("power hist"),
            L.hist(rho.flatten(), bins=100).title("rho hist"),
            L.scatter(rho.flatten(), power.flatten(), 1)
            .xlabel("rho")
            .ylabel("power")
            .title("power vs rho"),
            L.scatter(rho.flatten(), act.flatten(), 1)
            .xlabel("rho")
            .ylabel("act")
            .title("act vs rho"),
            L.hist(weig, bins=100).title("w/ eigs hist"),
        ],
        [
            L.hist(iact.flatten(), bins=100).title("act hist with fake weight"),
            L.hist(ipower.flatten(), bins=100).title("power hist with fake weight"),
            L.hist(irho.flatten(), bins=100).title("rho hist with fake weight"),
            L.scatter(irho.flatten(), ipower.flatten(), 1)
            .xlabel("rho")
            .ylabel("power")
            .title("power vs rho with fake weight"),
            L.scatter(irho.flatten(), iact.flatten(), 1)
            .xlabel("rho")
            .ylabel("act")
            .title("act vs rho with fake weight"),
            L.hist(iweig, bins=100).title("w/ eigs hist with fake weight"),
        ],
    ]

    rows = len(plots)
    cols = max(len(p) for p in plots)

    fig = plt.figure(figsize=(cols * 3, rows * 2))
    for i, row in enumerate(plots):
        for j, plot in enumerate(row):
            kwargs = {"projection": "polar"} if plot.polar else {}
            plt.subplot(rows, cols, i * cols + j + 1, **kwargs)
            plot.run()

    fig.suptitle(f"Inspecting {name}")
    plt.tight_layout()
    plt.show()


def inspect_rmsnorm(layer: torch.nn.Module, input: torch.Tensor, name: str = "RMSNorm"):
    # RMSNorm的实现
    weight = layer.weight
    input_dtype = input.dtype
    input = input.to(torch.float32)
    variance = input.pow(2).mean(-1, keepdim=True)
    output1 = input * torch.rsqrt(variance + layer.variance_epsilon)
    output2 = weight * output1.to(input_dtype)
    plots = [
        [
            L.hist(input.flatten(), bins=100).title("i/ hist"),
            L.hist(input.pow(2).sum(dim=-1).sqrt().flatten(), bins=100).title(
                "i/ token norm hist"
            ),
            L.hist(input.pow(2).sum(dim=1).sqrt().flatten(), bins=100).title(
                "i/ channel norm hist"
            ),
            L.hist(weight.flatten(), bins=100).title("w/ hist"),
        ],
        [
            L.hist(output1.flatten(), bins=100).title("o1/ hist"),
            L.hist(output1.pow(2).sum(dim=-1).sqrt().flatten(), bins=100).title(
                "o1/ token norm hist"
            ),
            L.hist(output1.pow(2).sum(dim=1).sqrt().flatten(), bins=100).title(
                "o1/ channel norm hist"
            ),
            L.hist(output2.flatten(), bins=100).title("o2/ hist"),
            L.hist(output2.pow(2).sum(dim=-1).sqrt().flatten(), bins=100).title(
                "o2/ token norm hist"
            ),
            L.hist(output2.pow(2).sum(dim=1).sqrt().flatten(), bins=100).title(
                "o2/ channel norm hist"
            ),
        ],
        [
            L.scatter(
                input.pow(2).sum(dim=-1).sqrt().flatten(),
                output1.pow(2).sum(dim=-1).sqrt().flatten(),
                1,
            )
            .xlabel("input")
            .ylabel("output1")
            .title("input vs output1 token norm"),
            L.scatter(
                input.pow(2).sum(dim=1).sqrt().flatten(),
                output1.pow(2).sum(dim=1).sqrt().flatten(),
                1,
            )
            .xlabel("input")
            .ylabel("output1")
            .title("input vs output1 channel norm"),
            L.scatter(
                output1.pow(2).sum(dim=-1).sqrt().flatten(),
                output2.pow(2).sum(dim=-1).sqrt().flatten(),
                1,
            )
            .xlabel("output1")
            .ylabel("output2")
            .title("output1 vs output2 token norm"),
            L.scatter(
                output1.pow(2).sum(dim=1).sqrt().flatten(),
                output2.pow(2).sum(dim=1).sqrt().flatten(),
                1,
            )
            .xlabel("output1")
            .ylabel("output2")
            .title("output1 vs output2 channel norm"),
            L.scatter(
                input.pow(2).sum(dim=-1).sqrt().flatten(),
                output2.pow(2).sum(dim=-1).sqrt().flatten(),
                1,
            )
            .xlabel("input")
            .ylabel("output2")
            .title("input vs output2 token norm"),
            L.scatter(
                input.pow(2).sum(dim=1).sqrt().flatten(),
                output2.pow(2).sum(dim=1).sqrt().flatten(),
                1,
            )
            .xlabel("input")
            .ylabel("output2")
            .title("input vs output2 channel norm"),
        ],
    ]
    rows = len(plots)
    cols = max(len(p) for p in plots)

    print(f"================== Inspecting {name} ==================")
    plt.figure(figsize=(cols * 3, rows * 2))
    for i, row in enumerate(plots):
        for j, plot in enumerate(row):
            kwargs = {"projection": "polar"} if plot.polar else {}
            plt.subplot(rows, cols, i * cols + j + 1, **kwargs)
            plot.run()
    plt.tight_layout()
    plt.show()
