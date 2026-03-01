import os
import json
import math
import time
import asyncio
from typing import Union, Literal, Optional, Iterator, List, Any, Dict
from tqdm import tqdm
import copy

from CARD.graph.graph import Graph
from experiments.accuracy import Accuracy
from CARD.utils.globals import Cost, PromptTokens, CompletionTokens, EdgesCount
import torch
from CARD.utils.const import CARD_ROOT


async def evaluate(
    graph: Graph,
    dataset,
    num_rounds: int = 1,
    limit_questions: Optional[int] = None,
    eval_batch_size: int = 4,
    eval_group: str = None,
) -> float:

    print(f"Evaluating CARD on {dataset.__class__.__name__} split {dataset.split}")

    graph.gcn.eval()
    graph.gcn_dynamic.eval()
    graph.feature_fusion.eval()
    graph.mlp.eval()

    model_path = f"{CARD_ROOT}/model_weights/mmlu/trained_gcn_beta0.005.pt"
    checkpoint = torch.load(model_path)
    graph.gcn.load_state_dict(checkpoint["gcn"])
    graph.gcn_dynamic.load_state_dict(checkpoint["gcn_dynamic"])
    graph.feature_fusion.load_state_dict(checkpoint["feature_fusion"])
    graph.mlp.load_state_dict(checkpoint["mlp"])
    print(f"Loaded model parameters from {model_path}")

    accuracy = Accuracy()

    def eval_loader(batch_size: int) -> Iterator[List[Any]]:
        records = []
        for i_record, record in enumerate(dataset):
            if limit_questions is not None:
                if i_record >= limit_questions:
                    break
            records.append(record)
            if len(records) >= batch_size:
                yield records
                records = []
        if len(records) > 0:
            yield records
        return

    data_len = (
        min(len(dataset), limit_questions)
        if limit_questions is not None
        else len(dataset)
    )
    num_batches = int(math.ceil(data_len / eval_batch_size))

    for i_batch, record_batch in tqdm(
        enumerate(eval_loader(batch_size=eval_batch_size)), total=num_batches
    ):
        print(80 * "-")

        start_ts = time.time()
        answer_log_probs = []

        for record in record_batch:
            realized_graph = copy.deepcopy(graph)
            realized_graph.gcn = graph.gcn
            realized_graph.mlp = graph.mlp
            realized_graph.gcn_dynamic = graph.gcn_dynamic
            realized_graph.feature_fusion = graph.feature_fusion
            input_dict = dataset.record_to_input(record)
            # print(input_dict)
            answer_log_probs.append(
                asyncio.create_task(
                    realized_graph.arun(input_dict, num_rounds, fixed_group=eval_group)
                )
            )
        raw_results = await asyncio.gather(*answer_log_probs)
        raw_answers, log_probs = zip(*raw_results)
        print(f"Batch time {time.time() - start_ts:.3f}")
        for raw_answer, record in zip(raw_answers, record_batch):
            print("Raw answer:", raw_answer)
            answer = dataset.postprocess_answer(raw_answer)
            print("Postprocessed answer:", answer)
            correct_answer = dataset.record_to_target_answer(record)
            print("Correct answer:", correct_answer)
            accuracy.update(answer, correct_answer)
            accuracy.print()
        print(f"Cost {Cost.instance().value}")
        print(f"PromptTokens {PromptTokens.instance().value}")
        print(f"CompletionTokens {CompletionTokens.instance().value}")
        print(f"EdgesCount {EdgesCount.instance().value}")

    accuracy.print()
    print("Done!")

    return accuracy.get()


def dump_eval_results(self, dct: Dict[str, Any]) -> None:
    if self._art_dir_name is not None:
        eval_json_name = os.path.join(self._art_dir_name, "evaluation.json")
        with open(eval_json_name, "w") as f:
            json.dump(dct, f)
