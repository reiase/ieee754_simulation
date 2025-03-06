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
        try:
            return self.fn(*self.args, **self.kwargs)
        except:
            import traceback

            traceback.print_exc()
            pass


class lazy:
    def __getattr__(self, name):
        plt_attr = getattr(plt, name)
        if callable(plt_attr):
            return delayed_call(plt_attr)
        return plt_attr


L = lazy()

def inspect_linear3(layer: torch.nn.Linear, input: torch.Tensor, name: str = "Linear"):
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

    plots = [
        [
            L.hist(input.flatten().cpu(), bins=100).title("i/ hist"),
            L.hist(weight.flatten().cpu(), bins=100).title("w/ hist"),
        ],
        [
            L.hist(act.flatten().cpu(), bins=100).title("act hist"),
            L.hist(power.flatten().cpu(), bins=100).title("power hist"),
            L.hist(rho.flatten().cpu(), bins=100).title("rho hist"),
        ],[
            L.scatter(rho.flatten().cpu(), power.flatten().cpu(), 1)
            .xlabel("rho")
            .ylabel("power")
            .title("power vs rho"),
            L.scatter(power.flatten().cpu(), act.flatten().cpu(), 1)
            .xlabel("power")
            .ylabel("act")
            .title("act vs power"),
            L.scatter(rho.flatten().cpu(), act.flatten().cpu(), 1)
            .xlabel("rho")
            .ylabel("act")
            .title("act vs rho"),
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


def inspect_linear2(layer: torch.nn.Linear, input: torch.Tensor, name: str = "Linear"):
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

    qweight = weight.to(torch.float8_e4m3fnuz).to(torch.float32)
    qinput = input.to(torch.float8_e4m3fnuz).to(torch.float32)

    qinput_norm = qinput.pow(2).sum(dim=2).sqrt()
    qweight_norm = qweight.pow(2).sum(dim=1).sqrt()

    qact = torch.matmul(qinput, qweight.T)
    qpower = torch.matmul(qinput_norm.T, qweight_norm.unsqueeze(dim=0))
    qrho = qact.squeeze(dim=0) / qpower

    # calculate the singular values of input, weight and activation
    ieig = torch.linalg.svdvals(input.squeeze(dim=0))
    weig = torch.linalg.svdvals(weight)
    qieig = torch.linalg.svdvals(qinput.squeeze(dim=0))
    qweig = torch.linalg.svdvals(qweight)

    plots = [
        [
            L.hist(input.flatten().cpu(), bins=100).title("i/ hist"),
            L.hist(input_norm.flatten().cpu(), bins=100).title("i/ token norm hist"),
            L.hist(input.pow(2).sum(dim=1).sqrt().flatten().cpu(), bins=100).title(
                "i/ channel norm hist"
            ),
            L.hist(ieig.cpu(), bins=100).title("i/ eigs hist"),
        ],
        [
            L.hist(weight.flatten().cpu(), bins=100).title("w/ hist"),
            L.hist(weight_norm.flatten().cpu(), bins=100).title("w/ token norm hist."),
            L.hist(weight.pow(2).sum(dim=0).sqrt().flatten().cpu(), bins=100).title(
                "w/ channel norm hist."
            ),
        ],
        [
            L.hist(qinput.flatten().cpu(), bins=100).title("qi/ hist"),
            L.hist(qinput_norm.flatten().cpu(), bins=100).title("qi/ token norm hist"),
            L.hist(qinput.pow(2).sum(dim=1).sqrt().flatten().cpu(), bins=100).title(
                "qi/ channel norm hist"
            ),
            L.hist(qieig.cpu(), bins=100).title("qi/ eigs hist"),
        ],
        [
            L.hist(qweight.flatten().cpu(), bins=100).title("qw/ hist"),
            L.hist(qweight_norm.flatten().cpu(), bins=100).title(
                "qw/ token norm hist."
            ),
            L.hist(qweight.pow(2).sum(dim=0).sqrt().flatten().cpu(), bins=100).title(
                "qw/ channel norm hist."
            ),
        ],
        [
            L.hist(act.flatten().cpu(), bins=100).title("act hist"),
            L.hist(power.flatten().cpu(), bins=100).title("power hist"),
            L.hist(rho.flatten().cpu(), bins=100).title("rho hist"),
            L.scatter(rho.flatten().cpu(), power.flatten().cpu(), 1)
            .xlabel("rho")
            .ylabel("power")
            .title("power vs rho"),
            L.scatter(rho.flatten().cpu(), act.flatten().cpu(), 1)
            .xlabel("rho")
            .ylabel("act")
            .title("act vs rho"),
            L.hist(weig.cpu(), bins=100).title("w/ eigs hist"),
        ],
        [
            L.hist(qact.flatten().cpu(), bins=100).title("qact hist"),
            L.hist(qpower.flatten().cpu(), bins=100).title("qpower hist"),
            L.hist(qrho.flatten().cpu(), bins=100).title("qrho hist"),
            L.scatter(qrho.flatten().cpu(), qpower.flatten().cpu(), 1)
            .xlabel("qrho")
            .ylabel("qpower")
            .title("qpower vs qrho"),
            L.scatter(qrho.flatten().cpu(), qact.flatten().cpu(), 1)
            .xlabel("qrho")
            .ylabel("qact")
            .title("qact vs qrho"),
            L.hist(qweig.cpu(), bins=100).title("qw/ eigs hist"),
        ],
        [
            L.hist((qact - act).flatten().cpu(), bins=100).title("act diff hist"),
            L.hist((qpower - power).flatten().cpu(), bins=100).title("power diff hist"),
            L.hist((qrho - rho).flatten().cpu(), bins=100).title("rho diff hist"),
            L.scatter((qpower - power).flatten().cpu(), (qact - act).flatten().cpu(), 1)
            .xlabel("qpower diff")
            .ylabel("qact diff")
            .title("qact diff vs qpower diff"),
            L.scatter((qrho - rho).flatten().cpu(), (qact - act).flatten().cpu(), 1)
            .xlabel("qrho diff")
            .ylabel("qact diff")
            .title("qact diff vs qrho diff"),
            L.hist((qweig - weig).cpu(), bins=100).title("qw/ eigs diff hist"),
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
    iweight = torch.normal(mean=0, std=0.02, size=weight.shape).cuda()
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
    
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    fig = plt.figure(figsize=(15, 5))
    outer_grid = GridSpec(1, 3, figure=fig)

    # 第一个联合分布图：input vs output1
    plt.subplot(1, 3, 1)
    scatter_with_histograms(
        input.flatten().cpu(),
        output1.flatten().cpu(),
        size=1,
        xlabel="input",
        ylabel="output1",
        title="input vs output1 with distributions",
        subplot_spec=outer_grid[0, 0]
    )

    # 第二个联合分布图：output1 vs output2
    plt.subplot(1, 3, 2)
    scatter_with_histograms(
        output1.flatten().cpu(),
        output2.flatten().cpu(),
        size=1,
        xlabel="output1",
        ylabel="output2",
        title="output1 vs output2 with distributions",
        subplot_spec=outer_grid[0, 1]
    )
    
    # 第三个联合分布图：input vs output2
    plt.subplot(1, 3, 3)
    scatter_with_histograms(
        input.flatten().cpu(),
        output2.flatten().cpu(),
        size=1,
        xlabel="input",
        ylabel="output2",
        title="input vs output2 with distributions",
        subplot_spec=outer_grid[0, 2]
    )
    
    # 添加总标题
    fig.suptitle(f"RMSNorm: {name}", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # 为总标题留出空间
    plt.show()


def scatter_with_histograms(
    x_data, y_data, size=1, xlabel="X", ylabel="Y", title="Joint Plot", subplot_spec=None
):
    """创建一个带有边缘直方图的散点图，支持嵌套在其他GridSpec中

    Args:
        x_data: x轴数据
        y_data: y轴数据
        size: 点的大小
        xlabel: x轴标签
        ylabel: y轴标签
        title: 图表标题
        subplot_spec: 父GridSpec中的子图规格，用于嵌套
    """
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    fig = plt.gcf()  # 获取当前图形
    fig.gca().axis("off")  # 关闭当前图形的坐标轴
    
    # 创建嵌套的GridSpec
    if subplot_spec is not None:
        gs = GridSpecFromSubplotSpec(5, 5, subplot_spec=subplot_spec)
    else:
        # 如果没有提供subplot_spec，就创建一个新图形和主GridSpec
        fig = plt.figure(figsize=(8, 8))
        gs = GridSpec(4, 4, figure=fig)
    
    # 主散点图
    ax_main = fig.add_subplot(gs[1:5, 0:4])
    ax_main.scatter(x_data, y_data, s=size)
    ax_main.set_xlabel(xlabel)
    ax_main.set_ylabel(ylabel)
    
    # x边缘直方图
    ax_x = fig.add_subplot(gs[0, 0:4], sharex=ax_main)
    ax_x.hist(x_data.numpy(), bins=100)
    ax_x.set_title(title)
    ax_x.tick_params(axis='x', labelbottom=False)
    
    # y边缘直方图
    ax_y = fig.add_subplot(gs[1:5, 4], sharey=ax_main)
    ax_y.hist(y_data.numpy(), bins=100, orientation='horizontal')
    ax_y.tick_params(axis='y', labelleft=False)
    
    return fig, (ax_main, ax_x, ax_y)

def inspect_swiglu(x, up_proj, gate_proj, act_gate_proj, down_proj, name):
    down_input = act_gate_proj * up_proj
    plots = [
        [
            L.hist(x.flatten().cpu(), bins=100).title("x hist"),
            L.hist(up_proj.flatten().cpu(), bins=100).title("up_proj hist"),
            L.hist(gate_proj.flatten().cpu(), bins=100).title("gate_proj hist"),
        ],
        [
            L.hist(act_gate_proj.flatten().cpu(), bins=100).title("act_gate_proj hist"),
            L.hist(down_input.flatten().cpu(), bins=100).title("down_input hist"),
            L.hist(down_proj.flatten().cpu(), bins=100).title("down_proj hist"),
        ],
        [
            L.scatter(up_proj.flatten().cpu(), gate_proj.flatten().cpu(), 1)
            .xlabel("up_proj")
            .ylabel("gate_proj")
            .title("gate_proj vs up_proj"),
            L.scatter(up_proj.flatten().cpu(), act_gate_proj.flatten().cpu(), 1)
            .xlabel("up_proj")
            .ylabel("act_gate_proj")
            .title("act_gate_proj vs up_proj"),
        ]
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