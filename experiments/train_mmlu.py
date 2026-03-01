import torch
from typing import Iterator
import pandas as pd
import numpy as np
import time
import asyncio
from typing import List
import copy

from CARD.graph.graph import Graph
from experiments.accuracy import Accuracy
from CARD.utils.globals import Cost, PromptTokens, CompletionTokens
from pathlib import Path
from CARD.utils.const import CARD_ROOT

prices_dict = {
    "gpt-4o-mini": 40,
    "deepseek-V3": 20,
    "llama-3-70B": 35,
    "gpt-4o": 60,
    "qwen-72B": 30,
}


async def train(
    graph: Graph,
    dataset,
    num_iters: int = 100,
    num_rounds: int = 1,
    lr: float = 0.1,
    batch_size: int = 4,
    eval_group: str = None,
) -> None:

    def infinite_data_loader() -> Iterator[pd.DataFrame]:
        perm = np.random.permutation(len(dataset))
        while True:
            for idx in perm:
                record = dataset[idx.item()]
                yield record

    loader = infinite_data_loader()

    graph.gcn.train()
    graph.gcn_dynamic.train()
    graph.feature_fusion.train()
    # graph.mlp.train()
    optimizer = torch.optim.Adam(
        list(graph.gcn.parameters())
        + list(graph.gcn_dynamic.parameters())
        + list(graph.feature_fusion.parameters()),
        lr=lr,
    )
    total_accuracy = Accuracy()

    for i_iter in range(num_iters):
        print(f"Iter {i_iter}", 80 * "-")
        start_ts = time.time()
        correct_answers = []
        answer_log_probs = []
        model_group = []

        for i_record, record in zip(range(batch_size), loader):
            realized_graph = copy.deepcopy(graph)
            realized_graph.gcn = graph.gcn
            realized_graph.gcn_dynamic = graph.gcn_dynamic
            realized_graph.feature_fusion = graph.feature_fusion
            realized_graph.mlp = graph.mlp
            input_dict = dataset.record_to_input(record)
            print(input_dict)
            if eval_group == "cycle":
                fixed_group = f"model_group_{i_record%5 + 1}"
                model_group.append(i_record % 5 + 1)
            elif eval_group == "iter":
                fixed_group = f"model_group_{i_iter%5 + 1}"
                model_group.append(i_iter % 5 + 1)
            else:
                fixed_group = None
            answer_log_probs.append(
                asyncio.create_task(
                    realized_graph.arun(input_dict, num_rounds, fixed_group=fixed_group)
                )
            )
            correct_answer = dataset.record_to_target_answer(record)
            correct_answers.append(correct_answer)

        raw_results = await asyncio.gather(*answer_log_probs)
        raw_answers, log_probs = zip(*raw_results)
        loss_list: List[torch.Tensor] = []
        utilities: List[float] = []
        answers: List[str] = []

        for model_num, raw_answer, log_prob, correct_answer in zip(
            model_group, raw_answers, log_probs, correct_answers
        ):
            answer = dataset.postprocess_answer(raw_answer)
            answers.append(answer)
            assert isinstance(
                correct_answer, str
            ), f"String expected but got {correct_answer} of type {type(correct_answer)} (1)"
            accuracy = Accuracy()
            accuracy.update(answer, correct_answer)
            total_accuracy.update(answer, correct_answer)
            utility = accuracy.get()
            utilities.append(utility)
            # Calculate token consumption penalty term

            avg_edge_price = list(prices_dict.values())[
                model_num - 1
            ]  # Custom average price consumption per edge
            edge_count = sum(
                torch.sigmoid(realized_graph.spatial_logits)
                * realized_graph.spatial_masks
            )
            price_penalty = avg_edge_price * edge_count

            # Combine accuracy loss and token consumption loss
            accuracy_loss = -log_prob * utility
            token_loss = 0.01 * price_penalty  # Use 0.01 as the weight coefficient for token loss
            single_loss = accuracy_loss + token_loss

            loss_list.append(single_loss)
            print(f"correct answer:{correct_answer}")

        total_loss = torch.mean(torch.stack(loss_list))
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print("raw_answers:", raw_answers)
        print("answers:", answers)
        print("total_accuracy:", total_accuracy.get())
        print(f"Batch time {time.time() - start_ts:.3f}")
        print("utilities:", utilities)  # [0.0, 0.0, 0.0, 1.0]
        print("loss:", total_loss.item())  # 4.6237263679504395
        print(f"Cost {Cost.instance().value}")
        print(f"PromptTokens {PromptTokens.instance().value}")
        print(f"CompletionTokens {CompletionTokens.instance().value}")

    model_path = Path(f"{CARD_ROOT}/model_weights/mmlu")
    model_path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "gcn": graph.gcn.state_dict(),
            "gcn_dynamic": graph.gcn_dynamic.state_dict(),
            "feature_fusion": graph.feature_fusion.state_dict(),
            "mlp": graph.mlp.state_dict(),
        },
        f"{CARD_ROOT}/model_weights/mmlu/trained_gcn_beta0.01.pt",
    )
    # f"{CARD_ROOT}/model_weights/mmlu/trained_gcn_{time.strftime('%Y%m%d_%H%M%S')}_{total_accuracy.get()}.pt")
