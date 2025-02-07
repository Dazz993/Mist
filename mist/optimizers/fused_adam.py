import traceback

import torch
import torch.distributed
import torch.distributed.pipeline
from torch.optim.adamw import _fused_adamw as torch_fused_adamw
from torch.optim.adamw import _single_tensor_adamw as torch_single_tensor_adamw
from apex.multi_tensor_apply import multi_tensor_applier

from mist.logger import get_logger

logger = get_logger()


class GradError(Exception):
    pass


class FusedAdam(torch.optim.Optimizer):
    """Implements Adam algorithm.

    Currently GPU-only.  Requires Apex to be installed via
    ``pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./``.

    This version of fused Adam implements 2 fusions.

      * Fusion of the Adam update's elementwise operations
      * A multi-tensor apply launch that batches the elementwise updates applied to all the model's parameters into one or a few kernel launches.

    :class:`apex.optimizers.FusedAdam` may be used as a drop-in replacement for ``torch.optim.AdamW``,
    or ``torch.optim.Adam`` with ``adam_w_mode=False``::

        opt = apex.optimizers.FusedAdam(model.parameters(), lr = ....)
        ...
        opt.step()

    :class:`apex.optimizers.FusedAdam` may be used with or without Amp.  If you wish to use :class:`FusedAdam` with Amp,
    you may choose any ``opt_level``::

        opt = apex.optimizers.FusedAdam(model.parameters(), lr = ....)
        model, opt = amp.initialize(model, opt, opt_level="O0" or "O1 or "O2")
        ...
        opt.step()

    In general, ``opt_level="O1"`` is recommended.


    .. warning::
        A previous version of :class:`FusedAdam` allowed a number of additional arguments to ``step``.  These additional arguments
        are now deprecated and unnecessary.

    Adam was been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square. (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False) NOT SUPPORTED in FusedAdam!
        adam_w_mode (boolean, optional): Apply L2 regularization or weight decay
            True for decoupled weight decay(also known as AdamW) (default: True)
        set_grad_none (bool, optional): whether set grad to None when zero_grad()
            method is called. (default: True)
        capturable (bool, optional): whether to use the version of the optimizer
            that can be used with CUDA Graphs. (default: False)
        master_weights (bool, optional): whether to maintain FP32 master weights
           in the optimizer with FP16 mixed precision training, currently can
           only be used with capturable set to True. (default: False)

    .. _Adam - A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        bias_correction=True,
        betas=(0.9, 0.999),
        eps=1e-8,
        adam_w_mode=True,
        weight_decay=0.0,
        amsgrad=False,
        set_grad_none=True,
        capturable=False,
        master_weights=False,
    ):
        if amsgrad:
            raise RuntimeError("FusedAdam does not support the AMSGrad variant.")
        if master_weights and not capturable:
            raise RuntimeError(
                "Master weights is currently only supported with the capturable version."
            )
        # If the optimizer is capturable then LR should be a tensor (on GPU)
        lr = torch.tensor(lr, dtype=torch.float32) if capturable else lr
        defaults = dict(
            lr=lr,
            bias_correction=bias_correction,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super(FusedAdam, self).__init__(params, defaults)
        self.adam_w_mode = 1 if adam_w_mode else 0
        self.set_grad_none = set_grad_none

        self.capturable = capturable
        self.master_weights = master_weights

        # Create full precision master weights
        self.param_groups_master = []
        for i, pg in enumerate(self.param_groups):
            param_list = pg["params"]
            self.param_groups_master.append(
                {
                    "params": [
                        p.clone().detach().float() if self.master_weights else None
                        for p in param_list
                    ],
                }
            )

        if capturable:
            for idx, group in enumerate(self.param_groups):
                if len(group["params"]) == 0:
                    continue
                device = group["params"][0].device
                for item in ["lr"]:
                    self.param_groups[idx][item] = group[item].to(device=device)

            self._step_supports_amp_scaling = True

        if multi_tensor_applier.available:
            import amp_C

            # Skip buffer
            self._dummy_overflow_buf = torch.cuda.IntTensor([0])
            self.multi_tensor_adam = amp_C.multi_tensor_adam
            self.multi_tensor_adam_capturable = amp_C.multi_tensor_adam_capturable
            self.multi_tensor_adam_capturable_master = (
                amp_C.multi_tensor_adam_capturable_master
            )
        else:
            raise RuntimeError("apex.optimizers.FusedAdam requires cuda extensions")

    def zero_grad(self):
        if self.set_grad_none:
            for group in self.param_groups:
                for p in group["params"]:
                    p.grad = None
        else:
            super(FusedAdam, self).zero_grad()

    def step(
        self,
        closure=None,
        grads=None,
        output_params=None,
        scale=None,
        grad_norms=None,
        grad_scaler=None,
    ):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.

        The remaining arguments are deprecated, and are only retained (for the moment) for error-checking purposes.
        """
        if any(p is not None for p in [grads, output_params, scale, grad_norms]):
            raise RuntimeError(
                "FusedAdam has been updated.  Simply initialize it identically to torch.optim.Adam, and call step() with no arguments."
            )
        loss = None
        if closure is not None:
            loss = closure()

        for group, group_master in zip(self.param_groups, self.param_groups_master):
            if len(group["params"]) == 0:
                continue
            device = group["params"][0].device
            bias_correction = 1 if group["bias_correction"] else 0
            beta1, beta2 = group["betas"]

            # assume same step across group now to simplify things
            # per parameter step can be easily support by making it tensor, or pass list into kernel
            if "step" in group:
                group["step"] += (
                    1
                    if not self.capturable
                    else (self._dummy_overflow_buf != 1).to(torch.int)
                )
            else:
                group["step"] = (
                    1
                    if not self.capturable
                    else torch.tensor([1], dtype=torch.int, device=device)
                )

            # create lists for multi-tensor apply
            g_16, p_16, m_16, v_16 = [], [], [], []
            g_bf, p_bf, m_bf, v_bf = [], [], [], []
            g_32, p_32, m_32, v_32 = [], [], [], []
            p_16_master = []
            p_32_master = []

            for p, p_master in zip(group["params"], group_master["params"]):
                if p.grad is None:
                    continue
                if p.grad.data.is_sparse:
                    raise RuntimeError(
                        "FusedAdam does not support sparse gradients, please consider SparseAdam instead"
                    )

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data).float()
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data).float()

                if p.dtype == torch.float16:
                    if self.master_weights:
                        p_16_master.append(p_master.data)
                    g_16.append(p.grad.data)
                    p_16.append(p.data)
                    m_16.append(state["exp_avg"])
                    v_16.append(state["exp_avg_sq"])
                elif p.dtype == torch.bfloat16:
                    g_bf.append(p.grad)
                    p_bf.append(p)
                    m_bf.append(state["exp_avg"])
                    v_bf.append(state["exp_avg_sq"])
                elif p.dtype == torch.float32:
                    if self.master_weights:
                        p_32_master.append(p_master.data)
                    g_32.append(p.grad.data)
                    p_32.append(p.data)
                    m_32.append(state["exp_avg"])
                    v_32.append(state["exp_avg_sq"])
                else:
                    raise RuntimeError("FusedAdam only support fp16 and fp32.")

            # If the optimizer is capturable, then if there's a grad scaler it works
            # on the GPU + a different multi_tensor_applier should be called
            if self.capturable:
                # overflow check of gradients
                found_inf = (
                    grad_scaler._check_inf_per_device(self)[device]
                    if grad_scaler is not None
                    else torch.zeros((1,), device=device)
                )
                self._dummy_overflow_buf.copy_(found_inf)

                # get unscale scale factor
                scale, inv_scale = None, None
                if grad_scaler:
                    scale = grad_scaler._get_scale_async()
                    inv_scale = scale.double().reciprocal().float()
                else:
                    scale = torch.ones((1,), device=device)
                    inv_scale = torch.ones((1,), device=device)

                if len(g_16) > 0:
                    multi_tensor_applier(
                        (
                            self.multi_tensor_adam_capturable_master
                            if self.master_weights
                            else self.multi_tensor_adam_capturable
                        ),
                        self._dummy_overflow_buf,
                        (
                            [g_16, p_16, m_16, v_16, p_16_master]
                            if self.master_weights
                            else [g_16, p_16, m_16, v_16]
                        ),
                        group["lr"],
                        beta1,
                        beta2,
                        group["eps"],
                        group["step"],
                        self.adam_w_mode,
                        bias_correction,
                        group["weight_decay"],
                        inv_scale,
                    )

                if len(g_bf) > 0:
                    multi_tensor_applier(
                        self.multi_tensor_adam_capturable,
                        self._dummy_overflow_buf,
                        [g_bf, p_bf, m_bf, v_bf],
                        group["lr"],
                        beta1,
                        beta2,
                        group["eps"],
                        group["step"],
                        self.adam_w_mode,
                        bias_correction,
                        group["weight_decay"],
                        inv_scale,
                    )

                if len(g_32) > 0:
                    multi_tensor_applier(
                        (
                            self.multi_tensor_adam_capturable_master
                            if self.master_weights
                            else self.multi_tensor_adam_capturable
                        ),
                        self._dummy_overflow_buf,
                        (
                            [g_32, p_32, m_32, v_32, p_32_master]
                            if self.master_weights
                            else [g_32, p_32, m_32, v_32]
                        ),
                        group["lr"],
                        beta1,
                        beta2,
                        group["eps"],
                        group["step"],
                        self.adam_w_mode,
                        bias_correction,
                        group["weight_decay"],
                        inv_scale,
                    )
            else:
                if len(g_16) > 0:
                    multi_tensor_applier(
                        self.multi_tensor_adam,
                        self._dummy_overflow_buf,
                        [g_16, p_16, m_16, v_16],
                        group["lr"],
                        beta1,
                        beta2,
                        group["eps"],
                        group["step"],
                        self.adam_w_mode,
                        bias_correction,
                        group["weight_decay"],
                    )

                if len(g_bf) > 0:
                    multi_tensor_applier(
                        self.multi_tensor_adam,
                        self._dummy_overflow_buf,
                        [g_bf, p_bf, m_bf, v_bf],
                        group["lr"],
                        beta1,
                        beta2,
                        group["eps"],
                        group["step"],
                        self.adam_w_mode,
                        bias_correction,
                        group["weight_decay"],
                    )

                if len(g_32) > 0:
                    multi_tensor_applier(
                        self.multi_tensor_adam,
                        self._dummy_overflow_buf,
                        [g_32, p_32, m_32, v_32],
                        group["lr"],
                        beta1,
                        beta2,
                        group["eps"],
                        group["step"],
                        self.adam_w_mode,
                        bias_correction,
                        group["weight_decay"],
                    )

        return loss


def fused_adamw_step(
    states,
    param_groups,
    param_groups_master=None,
    _dummy_overflow_buf=None,
    adam_w_mode=True,
    capturable=False,
    master_weights=False,
    closure=None,
    grads=None,
    output_params=None,
    scale=None,
    grad_norms=None,
    inv_scale=None,
    message=None,
):
    """Performs a single optimization step.

    Arguments:
        closure (callable, optional): A closure that reevaluates the model
            and returns the loss.

    The remaining arguments are deprecated, and are only retained (for the moment) for error-checking purposes.
    """
    if any(p is not None for p in [grads, output_params, scale, grad_norms]):
        raise RuntimeError(
            "FusedAdam has been updated.  Simply initialize it identically to torch.optim.Adam, and call step() with no arguments."
        )

    if master_weights is True:
        raise NotImplementedError(
            "FusedAdam does not support master weights with fused_adamw_step."
        )

    if multi_tensor_applier.available:
        import amp_C

        # Skip buffer
        multi_tensor_adam = amp_C.multi_tensor_adam
        multi_tensor_adam_capturable = amp_C.multi_tensor_adam_capturable
        multi_tensor_adam_capturable_master = amp_C.multi_tensor_adam_capturable_master
    else:
        raise RuntimeError("optimizers.FusedAdam requires cuda extensions")

    if param_groups_master is None:
        param_groups_master = []
        for i, pg in enumerate(param_groups):
            param_list = pg["params"]
            param_groups_master.append(
                {
                    "params": [
                        p.clone().detach().float() if master_weights else None
                        for p in param_list
                    ],
                }
            )

    if _dummy_overflow_buf is None:
        _dummy_overflow_buf = torch.cuda.IntTensor([0])

    loss = None
    if closure is not None:
        loss = closure()

    for group, group_master in zip(param_groups, param_groups_master):
        if len(group["params"]) == 0:
            continue
        device = group["params"][0].device
        bias_correction = 1 if group["bias_correction"] else 0
        beta1, beta2 = group["betas"]
        if capturable and not isinstance(group["lr"], torch.Tensor):
            group["lr"] = torch.tensor(group["lr"], dtype=torch.float32, device=device)

        # assume same step across group now to simplify things
        # per parameter step can be easily support by making it tensor, or pass list into kernel
        if "step" in group:
            group["step"] += (
                1 if not capturable else (_dummy_overflow_buf != 1).to(torch.int)
            )
        else:
            group["step"] = (
                1
                if not capturable
                else torch.tensor([1], dtype=torch.int, device=device)
            )

        # create lists for multi-tensor apply
        g_16, p_16, m_16, v_16 = [], [], [], []
        g_bf, p_bf, m_bf, v_bf = [], [], [], []
        g_32, p_32, m_32, v_32 = [], [], [], []
        p_16_master = []
        p_32_master = []

        for p, p_master in zip(group["params"], group_master["params"]):
            if p.grad is None:
                continue
            if p.grad.data.is_sparse:
                raise RuntimeError(
                    "FusedAdam does not support sparse gradients, please consider SparseAdam instead"
                )

            state = states[p]
            # State initialization
            if len(state) == 0:
                raise RuntimeError(
                    "In Mist, we hope state is initialized before calling fused_adamw_step."
                )
                # Exponential moving average of gradient values
                state["exp_avg"] = torch.zeros_like(p.data).float()
                # Exponential moving average of squared gradient values
                state["exp_avg_sq"] = torch.zeros_like(p.data).float()

            if p.dtype == torch.float16:
                if master_weights:
                    p_16_master.append(p_master.data)
                g_16.append(p.grad.data)
                p_16.append(p.data)
                m_16.append(state["exp_avg"].data)
                v_16.append(state["exp_avg_sq"].data)
            elif p.dtype == torch.bfloat16:
                g_bf.append(p.grad)
                p_bf.append(p)
                m_bf.append(state["exp_avg"])
                v_bf.append(state["exp_avg_sq"])
            elif p.dtype == torch.float32:
                if master_weights:
                    p_32_master.append(p_master.data)
                g_32.append(p.grad.data)
                # p_32.append(p.data.clone())
                p_32.append(p.data)
                m_32.append(state["exp_avg"].data)
                v_32.append(state["exp_avg_sq"].data)
            else:
                raise RuntimeError("FusedAdam only support fp16 and fp32.")

        # Check g_32
        # Get world size and rank
        # world_size = torch.distributed.get_world_size()
        # rank = torch.distributed.get_rank()
        # assert g_32[0].numel() % world_size == 0
        # chunk_size = g_32[0].numel() // world_size
        # for gg, pp, mm, vv in zip(g_32, p_32, m_32, v_32):
        #     gg_chunk = gg.flatten()[rank * chunk_size : (rank + 1) * chunk_size]
        #     pp_chunk = pp.flatten()[rank * chunk_size : (rank + 1) * chunk_size]
        #     mm_chunk = mm.flatten()[rank * chunk_size : (rank + 1) * chunk_size]
        #     vv_chunk = vv.flatten()[rank * chunk_size : (rank + 1) * chunk_size]
        #     logger.error(
        #         f"[{message}] Whole size: {gg.flatten().numel()}, {pp.flatten().numel()}, {mm.flatten().numel()}, {vv.flatten().numel()}"
        #     )
        #     logger.error(
        #         f"[{message}] Before step (chunk): {gg_chunk.float().abs().sum():.8f}, {pp_chunk.float().abs().sum():.8f}, {mm_chunk.float().abs().sum():.8f}, {vv_chunk.float().abs().sum():.8f}"
        #     )
        #     logger.error(
        #         f"[{message}] Before step (whole): {gg.flatten().float().abs().sum():.8f}, {pp.flatten().float().abs().sum():.8f}, {mm.flatten().float().abs().sum():.8f}, {vv.flatten().float().abs().sum():.8f}"
        #     )

        # Check inf and zeros in the gradients
        for grad in g_32:
            checked_num = grad.numel() // 2  # Should use 1/3 for fair comparison, but 1/3 may not be good for computation efficiency
            checked_grad = grad.flatten()[:checked_num]

            # Chunked check
            chunk_size = 8 * 1024 ** 2
            num_chunks = (checked_num + chunk_size - 1) // chunk_size
            num_nonzeros = 0
            for i in range(num_chunks):
                chunk = checked_grad[i * chunk_size: min((i + 1) * chunk_size, checked_num)]
                if i == 0:
                    if torch.isinf(chunk).any():
                        raise GradError("FusedAdam does not support inf in gradients.")
                    if torch.isnan(chunk).any():
                        raise GradError("FusedAdam does not support nan in gradients.")
                num_nonzeros += torch.count_nonzero(chunk)
            if num_nonzeros == 0:
                raise GradError(
                    f"Zero grads will make the benchmarking invalid. Check {grad.shape=}."
                )
            logger.info(f"[{message}] Number Zeros Ratio: {1 - num_nonzeros / checked_num:.2f}.")

        # torch.cuda.synchronize()
        # for i, (gg, pp, mm, vv) in enumerate(zip(g_32, p_32, m_32, v_32)):
        #     logger.error(
        #         f"[{message=}] Before {i=}, g={gg.flatten().sum():.8f}, p={pp.flatten().sum():.8f}, m={mm.flatten().sum():.8f}, v={vv.flatten().sum():.8f}"
        #     )

        # If the optimizer is capturable, then if there's a grad scaler it works
        # on the GPU + a different multi_tensor_applier should be called
        if capturable:
            # overflow check of gradients
            found_inf = torch.zeros((1,), device=device)
            _dummy_overflow_buf.copy_(found_inf)

            # get unscale scale factor
            if inv_scale is None:
                inv_scale = torch.ones((1,), device=device)

            if len(g_16) > 0:
                multi_tensor_applier(
                    (
                        multi_tensor_adam_capturable_master
                        if master_weights
                        else multi_tensor_adam_capturable
                    ),
                    _dummy_overflow_buf,
                    (
                        [g_16, p_16, m_16, v_16, p_16_master]
                        if master_weights
                        else [g_16, p_16, m_16, v_16]
                    ),
                    group["lr"],
                    beta1,
                    beta2,
                    group["eps"],
                    group["step"],
                    adam_w_mode,
                    bias_correction,
                    group["weight_decay"],
                    inv_scale,
                )

            if len(g_bf) > 0:
                multi_tensor_applier(
                    multi_tensor_adam_capturable,
                    _dummy_overflow_buf,
                    [g_bf, p_bf, m_bf, v_bf],
                    group["lr"],
                    beta1,
                    beta2,
                    group["eps"],
                    group["step"],
                    adam_w_mode,
                    bias_correction,
                    group["weight_decay"],
                    inv_scale,
                )

            if len(g_32) > 0:
                multi_tensor_applier(
                    (
                        multi_tensor_adam_capturable_master
                        if master_weights
                        else multi_tensor_adam_capturable
                    ),
                    _dummy_overflow_buf,
                    (
                        [g_32, p_32, m_32, v_32, p_32_master]
                        if master_weights
                        else [g_32, p_32, m_32, v_32]
                    ),
                    group["lr"],
                    beta1,
                    beta2,
                    group["eps"],
                    group["step"],
                    adam_w_mode,
                    bias_correction,
                    group["weight_decay"],
                    inv_scale,
                )
        else:
            if len(g_16) > 0:
                multi_tensor_applier(
                    multi_tensor_adam,
                    _dummy_overflow_buf,
                    [g_16, p_16, m_16, v_16],
                    group["lr"],
                    beta1,
                    beta2,
                    group["eps"],
                    group["step"],
                    adam_w_mode,
                    bias_correction,
                    group["weight_decay"],
                )

            if len(g_bf) > 0:
                multi_tensor_applier(
                    multi_tensor_adam,
                    _dummy_overflow_buf,
                    [g_bf, p_bf, m_bf, v_bf],
                    group["lr"],
                    beta1,
                    beta2,
                    group["eps"],
                    group["step"],
                    adam_w_mode,
                    bias_correction,
                    group["weight_decay"],
                )

            if len(g_32) > 0:
                # print(f"[RANK={torch.distributed.get_rank()}] Before step: {g_32[0]=} {p_32[0]=} {m_32[0]=} {v_32[0]=}")
                multi_tensor_applier(
                    multi_tensor_adam,
                    _dummy_overflow_buf,
                    [g_32, p_32, m_32, v_32],
                    group["lr"],
                    beta1,
                    beta2,
                    group["eps"],
                    group["step"],
                    adam_w_mode,
                    bias_correction,
                    group["weight_decay"],
                )

                # max_exp_avg_sqs = []
                # state_steps = [
                #     (
                #         torch.tensor(group["step"], dtype=torch.float32, device=device)
                #         if not isinstance(group["step"], torch.Tensor)
                #         else group["step"].to(device=device)
                #     )
                #     for p in group["params"]
                # ]
                # grad_scale = (
                #     torch.tensor(inv_scale, device=device)
                #     if not isinstance(inv_scale, torch.Tensor)
                #     else inv_scale.to(device=device)
                # )
                # grad_scale = 1.0 / grad_scale
                # found_inf = None
                # torch_fused_adamw(
                #     p_32,
                #     g_32,
                #     m_32,
                #     v_32,
                #     max_exp_avg_sqs,
                #     state_steps,
                #     grad_scale,
                #     found_inf,
                #     amsgrad=False,
                #     beta1=beta1,
                #     beta2=beta2,
                #     lr=group["lr"],
                #     weight_decay=group["weight_decay"],
                #     eps=group["eps"],
                #     maximize=False,
                #     capturable=False,
                #     differentiable=False,
                # )

                # torch_single_tensor_adamw(
                #     p_32,
                #     g_32,
                #     m_32,
                #     v_32,
                #     max_exp_avg_sqs,
                #     state_steps,
                #     None,
                #     None,
                #     amsgrad=False,
                #     beta1=beta1,
                #     beta2=beta2,
                #     lr=group["lr"],
                #     weight_decay=group["weight_decay"],
                #     eps=group["eps"],
                #     maximize=False,
                #     capturable=False,
                #     differentiable=False,
                # )

    # torch.cuda.synchronize()
    # for i, (gg, pp, mm, vv) in enumerate(zip(g_32, p_32, m_32, v_32)):
    #     # checked_num = min(128, gg.numel())
    #     # checked_slice = slice(-checked_num, None)
    #     checked_slice = slice(None)
    #     logger.error(
    #         f"[{message=}] After {i=}, g={gg.flatten()[checked_slice].sum():.8f}, p={pp.flatten()[checked_slice].sum():.8f}, "
    #         f"m={mm.flatten()[checked_slice].sum():.8f}, v={vv.flatten()[checked_slice].sum():.8f}"
    #     )

    return loss
