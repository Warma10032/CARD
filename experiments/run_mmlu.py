import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

import asyncio
from typing import Union, Literal, List
import argparse
import random
from io import StringIO
import json
from datetime import datetime


from CARD.graph.graph import Graph
from datasets.mmlu_dataset import MMLUDataset
from datasets.MMLU.download import download
from experiments.train_mmlu import train
from experiments.evaluate_mmlu import evaluate
from CARD.utils.const import CARD_ROOT

import debugpy

try:
    # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
    debugpy.listen(("localhost", 9301))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
except Exception as e:
    pass


def parse_args():
    parser = argparse.ArgumentParser(description="Process some parameters.")

    parser.add_argument(
        "--mode",
        type=str,
        default="FullConnected",
        choices=[
            "DirectAnswer",
            "FullConnected",
            "Random",
            "Chain",
            "Debate",
            "Layered",
            "Star",
            "Mesh",
            "FakeFullConnected",
            "FakeRandom",
            "FakeChain",
            "FakeStar",
            "FakeMesh",
            "FakeAGRandom",
            "FakeAGFull",
        ],
        help="Mode of operation. Default is 'FullConnected'.",
    )
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument(
        "--agent_names",
        nargs="+",
        type=str,
        default=["AnalyzeAgent"],
        help="Specify agent names as a list of strings",
    )
    parser.add_argument(
        "--agent_nums",
        nargs="+",
        type=int,
        default=[5],
        help="Specify the number of agents for each name in agent_names",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=10,
        help="Number of optimization iterations. Default 10.",
    )
    parser.add_argument(
        "--imp_per_iterations",
        type=int,
        default=5,
        help="Prune every few iterations. Default 5.",
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=1,
        help="Number of optimization/inference rounds for one query",
    )
    parser.add_argument(
        "--pruning_rate",
        type=float,
        default=0.25,
        help="The Rate of Pruning. Default 0.05.",
    )
    parser.add_argument(
        "--llm_name",
        type=str,
        default="gpt-4o-mini",
        help="Model name, None runs the default ChatGPT4",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="mmlu",
        help="Domain (the same as dataset name), default 'MMLU'",
    )
    parser.add_argument(
        "--decision_method",
        type=str,
        default="FinalRefer",
        help="the decision method of the final node",
    )
    parser.add_argument("--optimized_spatial", action="store_true")
    parser.add_argument("--optimized_temporal", action="store_true")
    parser.add_argument(
        "--node_config_file",
        type=str,
        default=".\CARD\config\mmlu_node_config.json",
        help="Path to JSON file containing node configurations.",
    )
    parser.add_argument("--phase", choices=["train", "eval", "test"], required=True)
    parser.add_argument(
        "--eval_group", type=str, default="train_group", help="Fixed combination name for evaluation phase"
    )

    args = parser.parse_args()
    result_path = CARD_ROOT / "result"
    os.makedirs(result_path, exist_ok=True)
    if len(args.agent_names) != len(args.agent_nums):
        parser.error("The number of agent names must match the number of agent counts.")

    return args


class ConsoleLogger:
    def __init__(self, log_file: str):
        self.original_stdout = sys.stdout
        self.log_buffer = StringIO()
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # Create dual output stream (output to both terminal and buffer)
        self.dual_output = Tee(self.original_stdout, self.log_buffer)

    def __enter__(self):
        sys.stdout = self.dual_output
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        self.save_logs()

    def save_logs(self):
        try:
            # Get and format log content
            log_content = self.log_buffer.getvalue()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Use more readable log format
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] LOG START\n")
                f.write(log_content)
                f.write(f"[{timestamp}] LOG END\n\n")
        except Exception as e:
            print(f"Failed to save logs: {str(e)}", file=self.original_stdout)


class Tee:
    """Dual output stream proxy class for real-time output"""

    def __init__(self, *outputs):
        self.outputs = outputs

    def write(self, text):
        for output in self.outputs:
            output.write(text)
            output.flush()  # Ensure real-time output

    def flush(self):
        for output in self.outputs:
            output.flush()


async def main():
    import time

    start_time = time.time()

    args = parse_args()

    log_dir = CARD_ROOT / "result" / "mmlu"
    log_dir.mkdir(exist_ok=True)
    log_file = (
        log_dir
        / f"mmlu_{args.phase}_{args.eval_group}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    # Keep the context manager usage unchanged
    with ConsoleLogger(str(log_file)) as logger:
        print(args)
        mode = args.mode
        decision_method = args.decision_method
        agent_names = [
            name
            for name, num in zip(args.agent_names, args.agent_nums)
            for _ in range(num)
        ]
        kwargs = get_kwargs(mode, len(agent_names))

        limit_questions = 153
        node_config = get_node_config(args.node_config_file, len(agent_names))

        download()

        if args.phase == "train":
            dataset_train = MMLUDataset("dev")
            graph = Graph(
                domain=args.domain,
                llm_name=args.llm_name,
                agent_names=agent_names,
                decision_method=decision_method,
                optimized_spatial=args.optimized_spatial,
                optimized_temporal=args.optimized_temporal,
                node_kwargs=node_config,
                allow_random_combination=False,
                **kwargs,
            )
            await train(
                graph=graph,
                dataset=dataset_train,
                num_iters=args.num_iterations,
                num_rounds=args.num_rounds,
                lr=args.lr,
                batch_size=args.batch_size,
                eval_group=args.eval_group,
            )
        elif args.phase == "eval":
            dataset_val = MMLUDataset("val")
            graph = Graph(
                domain=args.domain,
                llm_name=args.llm_name,
                agent_names=agent_names,
                decision_method=decision_method,
                optimized_spatial=args.optimized_spatial,
                optimized_temporal=args.optimized_temporal,
                node_kwargs=node_config,
                allow_random_combination=False,
                **kwargs,
            )
            score = await evaluate(
                graph=graph,
                dataset=dataset_val,
                num_rounds=args.num_rounds,
                limit_questions=limit_questions,
                eval_batch_size=args.batch_size,
                eval_group=args.eval_group,
            )
            print(f"Score: {score}")  # Now displays in real-time on terminal

        # Record total execution time
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")


def get_node_config(config_file: str, agents_num: int) -> dict:
    with open(config_file, "r", encoding="utf-8") as f:
        all_node_configs = json.load(f)

    for combo_name, config_list in all_node_configs.items():
        if not isinstance(config_list, list):
            raise TypeError(f"Configuration for {combo_name} is not a list type")
        assert (
            len(config_list) == agents_num
        ), f"Number of nodes for {combo_name} is {len(config_list)}, but expected {agents_num}"

    return all_node_configs


def get_kwargs(
    mode: Union[
        Literal["DirectAnswer"],
        Literal["FullConnected"],
        Literal["Random"],
        Literal["Chain"],
        Literal["Debate"],
        Literal["Layered"],
        Literal["Star"],
        Literal["Mesh"],
        Literal["FakeFullConnected"],
        Literal["FakeRandom"],
        Literal["FakeChain"],
        Literal["FakeStar"],
        Literal["FakeMesh"],
        Literal["FakeAGRandom"],
        Literal["FakeAGFull"],
    ],
    N: int,
):
    initial_spatial_probability: float = 0.5
    fixed_spatial_masks: List[List[int]] = None
    initial_temporal_probability: float = 0.5
    fixed_temporal_masks: List[List[int]] = None
    node_kwargs = None

    def generate_layered_graph(N, layer_num=2):
        adj_matrix = [[0] * N for _ in range(N)]
        base_size = N // layer_num
        remainder = N % layer_num
        layers = []
        for i in range(layer_num):
            size = base_size + (1 if i < remainder else 0)
            layers.extend([i] * size)
        random.shuffle(layers)
        for i in range(N):
            current_layer = layers[i]
            for j in range(N):
                if layers[j] == current_layer + 1:
                    adj_matrix[i][j] = 1
        return adj_matrix

    def generate_mesh_graph(N):
        adj_matrix = [[0] * N for _ in range(N)]
        for i in range(0, N):
            for j in range(i + 1, N):
                adj_matrix[i][j] = 1
        return adj_matrix

    def generate_star_graph(N):
        adj_matrix = [[0] * N for _ in range(N)]
        for i in range(1, N):
            adj_matrix[0][i] = 1
        return adj_matrix

    if mode == "DirectAnswer":
        fixed_spatial_masks = [[0]]
        fixed_temporal_masks = [[0]]
        node_kwargs = [{"role": "Normal"}]
    elif mode == "FullConnected" or mode == "FakeFullConnected" or mode == "FakeAGFull":
        fixed_spatial_masks = [[1 if i != j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode == "Random" or mode == "FakeRandom" or mode == "FakeAGRandom":
        fixed_spatial_masks = [
            [random.randint(0, 1) if i != j else 0 for i in range(N)] for j in range(N)
        ]
        fixed_temporal_masks = [
            [random.randint(0, 1) for _ in range(N)] for _ in range(N)
        ]
    elif mode == "Chain" or mode == "FakeChain":
        fixed_spatial_masks = [
            [1 if i == j + 1 else 0 for i in range(N)] for j in range(N)
        ]
        fixed_temporal_masks = [
            [1 if i == 0 and j == N - 1 else 0 for i in range(N)] for j in range(N)
        ]
    elif mode == "Debate":
        fixed_spatial_masks = [[0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == "Layered":
        fixed_spatial_masks = generate_layered_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == "Mesh" or mode == "FakeMesh":
        fixed_spatial_masks = generate_mesh_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == "Star" or mode == "FakeStar":
        fixed_spatial_masks = generate_star_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]

    if "Fake" in mode and "AG" not in mode:
        node_kwargs = [
            {"role": "Fake"} if i % 2 == N % 2 else {"role": "Normal"} for i in range(N)
        ]
    elif "Fake" in mode and "AG" in mode:
        node_kwargs = [
            {"role": "Fake"} if i % 2 == N % 2 else {"role": None} for i in range(N)
        ]

    return {
        "initial_spatial_probability": initial_spatial_probability,
        "fixed_spatial_masks": fixed_spatial_masks,
        "initial_temporal_probability": initial_temporal_probability,
        "fixed_temporal_masks": fixed_temporal_masks,
    }


if __name__ == "__main__":
    asyncio.run(main())
