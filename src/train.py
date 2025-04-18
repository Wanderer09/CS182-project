import os
from random import randint
import uuid

from quinine import QuinineArgumentParser
from tqdm import tqdm
import torch
import yaml

from eval import get_run_metrics
from tasks import get_task_sampler
from samplers import get_data_sampler
from curriculum import Curriculum
from schema import schema
from models import build_model

import wandb

# 启用 cuDNN 的自动优化以加速训练
torch.backends.cudnn.benchmark = True


def train_step(model, xs, ys, optimizer, loss_func):
    """
    执行单步训练：
    - 前向传播计算输出
    - 计算损失并反向传播
    - 更新模型参数
    返回损失值和模型输出
    """
    optimizer.zero_grad()  # 清空优化器的梯度
    output = model(xs, ys)  # 前向传播计算模型输出
    loss = loss_func(output, ys)  # 计算损失
    loss.backward()  # 反向传播计算梯度
    optimizer.step()  # 更新模型参数
    return loss.detach().item(), output.detach()  # 返回损失值和模型输出（从计算图中分离）


def sample_seeds(total_seeds, count):
    """
    随机生成一组种子，用于数据采样。
    - total_seeds: 总种子数
    - count: 需要生成的种子数量
    返回生成的种子集合。
    """
    seeds = set()
    while len(seeds) < count:  # 循环生成直到达到所需数量
        seeds.add(randint(0, total_seeds - 1))
    return seeds


def train(model, args):
    """
    执行完整的训练流程：
    - 初始化优化器和课程学习
    - 支持从断点恢复训练
    - 动态采样数据和任务
    - 记录训练指标并保存模型
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)  # 初始化优化器
    curriculum = Curriculum(args.training.curriculum)  # 初始化课程学习

    # 检查是否需要从断点恢复
    starting_step = 0
    state_path = os.path.join(args.out_dir, "state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path)  # 加载保存的训练状态
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]
        for i in range(state["train_step"] + 1):
            curriculum.update()  # 更新课程学习的状态

    # 获取模型的维度信息
    n_dims = model.n_dims
    bsize = args.training.batch_size
    # 初始化数据采样器和任务采样器
    data_sampler = get_data_sampler(args.training.data, n_dims=n_dims)
    task_sampler = get_task_sampler(
        args.training.task,
        n_dims,
        bsize,
        num_tasks=args.training.num_tasks,
        **args.training.task_kwargs,
    )
    pbar = tqdm(range(starting_step, args.training.train_steps))  # 初始化进度条

    num_training_examples = args.training.num_training_examples

    for i in pbar:
        # 动态调整采样参数
        data_sampler_args = {}
        task_sampler_args = {}

        if "sparse" in args.training.task:
            task_sampler_args["valid_coords"] = curriculum.n_dims_truncated
        if num_training_examples is not None:
            assert num_training_examples >= bsize
            seeds = sample_seeds(num_training_examples, bsize)
            data_sampler_args["seeds"] = seeds
            task_sampler_args["seeds"] = [s + 1 for s in seeds]

        # 采样输入数据
        xs = data_sampler.sample_xs(
            curriculum.n_points,
            bsize,
            curriculum.n_dims_truncated,
            **data_sampler_args,
        )
        # 采样任务并计算目标值
        task = task_sampler(**task_sampler_args)
        ys = task.evaluate(xs)

        # 获取损失函数
        loss_func = task.get_training_metric()

        # 执行单步训练
        loss, output = train_step(model, xs.cuda(), ys.cuda(), optimizer, loss_func)

        # 计算点级别的损失
        point_wise_tags = list(range(curriculum.n_points))
        point_wise_loss_func = task.get_metric()
        point_wise_loss = point_wise_loss_func(output, ys.cuda()).mean(dim=0)

        # 计算基线损失
        baseline_loss = (
            sum(
                max(curriculum.n_dims_truncated - ii, 0)
                for ii in range(curriculum.n_points)
            )
            / curriculum.n_points
        )

        # 记录训练指标到 wandb
        if i % args.wandb.log_every_steps == 0 and not args.test_run:
            wandb.log(
                {
                    "overall_loss": loss,
                    "excess_loss": loss / baseline_loss,
                    "pointwise/loss": dict(
                        zip(point_wise_tags, point_wise_loss.cpu().numpy())
                    ),
                    "n_points": curriculum.n_points,
                    "n_dims": curriculum.n_dims_truncated,
                },
                step=i,
            )

        curriculum.update()  # 更新课程学习状态

        pbar.set_description(f"loss {loss}")  # 更新进度条描述
        # 定期保存训练状态
        if i % args.training.save_every_steps == 0 and not args.test_run:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
            }
            torch.save(training_state, state_path)

        # 定期保存模型快照
        if (
            args.training.keep_every_steps > 0
            and i % args.training.keep_every_steps == 0
            and not args.test_run
            and i > 0
        ):
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_{i}.pt"))


def main(args):
    """
    训练脚本的入口函数：
    - 初始化 wandb（如果不是测试运行）
    - 构建模型并设置为训练模式
    - 调用 train 函数执行训练
    - 在非测试运行中预计算评估指标
    """
    if args.test_run:
        # 测试运行时缩短课程学习的范围
        curriculum_args = args.training.curriculum
        curriculum_args.points.start = curriculum_args.points.end
        curriculum_args.dims.start = curriculum_args.dims.end
        args.training.train_steps = 100
    else:
        # 初始化 wandb
        wandb.init(
            dir=args.out_dir,
            project=args.wandb.project,
            entity=args.wandb.entity,
            config=args.__dict__,
            notes=args.wandb.notes,
            name=args.wandb.name,
            resume=True,
        )

    # 构建模型并设置为训练模式
    model = build_model(args.model)
    model.cuda()
    model.train()

    # 调用训练函数
    train(model, args)

    # 非测试运行时预计算评估指标
    if not args.test_run:
        _ = get_run_metrics(args.out_dir)  # 预计算评估指标


if __name__ == "__main__":
    """
    脚本入口：
    - 解析配置文件
    - 验证模型类型
    - 设置输出目录并保存配置
    - 调用 main 函数开始训练
    """
    parser = QuinineArgumentParser(schema=schema)  # 解析配置文件
    args = parser.parse_quinfig()
    assert args.model.family in ["gpt2", "lstm"]  # 验证模型类型
    print(f"Running with: {args}")

    if not args.test_run:
        # 设置输出目录
        run_id = args.training.resume_id
        if run_id is None:
            run_id = str(uuid.uuid4())

        out_dir = os.path.join(args.out_dir, run_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir

        # 保存配置到文件
        with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    main(args)  # 调用主函数